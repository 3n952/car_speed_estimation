from pytube import YouTube
import argparse

# arguments 설정
def parse_arguments():
    parser = argparse.ArgumentParser(
        description= 'youtube video download'
    )
    parser.add_argument(
        "--url", required= True,
        help = "url to the youtube video",
        type = str,
    )
    parser.add_argument(
        "--save_dir", default= r'.\data',
        help = "directory where youtube video save",
        type = str,
    )
    
    return parser.parse_args()

def download_youtube_video(url, output_path):
    
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=output_path)
        print(f"Downloaded: {yt.title}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__=="__main__":
    args = parse_arguments()
    download_youtube_video(args.url, args.save_dir)


