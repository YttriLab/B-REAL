"""Reading a learning-rate range test (training_service.find_lr)."""


def valley_lr(lrs, losses):
    """fastai's "valley" rule: the longest stretch over which the loss keeps
    falling, and a learning rate a little past its middle. Lightning's own
    suggestion takes the steepest slope against the *linear* learning rate, so
    a wobble at 1e-6 (where the spacing is tiny) outranks the real descent a few
    decades up; on rig2 it picked 3.5e-6 off a curve that falls from 1e-4 to
    2e-3. Returns None when nothing falls."""
    n = len(losses)
    if n < 3:
        return None
    run = [1] * n  # longest decreasing subsequence ending at each point
    best_start = best_end = 0
    for i in range(1, n):
        for j in range(i):
            if losses[i] < losses[j] and run[i] < run[j] + 1:
                run[i] = run[j] + 1
        if run[best_end] < run[i]:
            best_end = i
            best_start = best_end - run[best_end]
    if best_end <= best_start:
        return None
    third = (best_end - best_start) / 3
    return float(lrs[best_start + int(third) + int(third / 2)])
