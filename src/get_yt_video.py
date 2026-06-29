from yt_dlp import YoutubeDL

def get_yt_video_link(query):
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(f"ytsearch3:{query}", download=False)

    video_titles = []
    video_links = []

    for video in results.get("entries", []):
        if video.get("id"):
            video_titles.append(video.get("title", "No Title"))
            video_links.append(f"https://www.youtube.com/watch?v={video['id']}")

    return video_titles, video_links