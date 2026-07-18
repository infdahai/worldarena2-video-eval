def output_filename(episode_id: int) -> str:
    """Return the traceable filename recommended by the Track 1 guide."""
    if episode_id <= 0:
        raise ValueError("episode_id must be positive")
    return f"episode_{episode_id:06d}.mp4"
