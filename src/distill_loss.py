"""Loss functions used by response and logit knowledge distillation.

The functions in this module are deliberately framework-light: importing the
pipeline does not require PyTorch until a loss is actually computed.
"""


def masked_kl_div(student_logits, teacher_logits, attention_mask=None, temperature=2.0):
    """Return teacher||student KL divergence, optionally over non-padding tokens."""
    import torch

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape; "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    student = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher = torch.softmax(teacher_logits / temperature, dim=-1)
    loss = torch.sum(teacher * (torch.log(teacher.clamp_min(1e-8)) - student), dim=-1)
    if attention_mask is not None:
        mask = attention_mask.to(dtype=loss.dtype)
        if mask.shape != loss.shape:
            raise ValueError(
                f"attention_mask shape {tuple(mask.shape)} does not match loss shape {tuple(loss.shape)}"
            )
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
    else:
        loss = loss.mean()
    return loss * (temperature ** 2)


def distillation_loss(
    student_logits,
    teacher_logits,
    labels=None,
    attention_mask=None,
    temperature=2.0,
    alpha=0.5,
):
    """Combine soft-target KL loss with ordinary next-token cross entropy."""
    import torch
    import torch.nn.functional as F

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    soft = masked_kl_div(student_logits, teacher_logits, attention_mask, temperature)
    if labels is None:
        return soft
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("student_logits must be [batch, sequence, vocab] and labels [batch, sequence]")
    hard = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return alpha * soft + (1 - alpha) * hard