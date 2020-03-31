"""Example of how to use the API."""
import argparse
import requests

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCI NDR Analysis.')
    parser.add_argument(
        'url', type=str, help='url to send requests.')
    parser.add_argument(
        'image_path', type=str, help='path to .png to analyze.')
    args = parser.parse_args()

    result = requests.post(
        args.url,
        data=open(args.image_path, 'rb').read(),
        headers={'Content-Type': 'image/png'})

    print(result)
