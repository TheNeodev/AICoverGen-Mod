from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Base URL for downloading
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

# Determine the base directory (adjust as needed)
BASE_DIR = Path(__file__).resolve().parent.parent
rvc_models_dir = BASE_DIR / 'rvc_models'
rvc_models_dir.mkdir(parents=True, exist_ok=True)

def get_model_links(url, extensions=('.pt',)):
    """
    Fetch the HTML at the given URL and use BeautifulSoup to find all links
    that end with the provided extensions (default: .pt files).
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all anchor tags with href attributes and filter by extension
    model_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if any(href.endswith(ext) for ext in extensions):
            model_links.append(href)
    return model_links

def dl_model(base_url, model_name, save_dir):
    """
    Download a model file from base_url+model_name to the specified directory.
    Uses tqdm to display a realtime progress bar.
    """
    download_url = f'{base_url}{model_name}'
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 8192

        # Initialize tqdm progress bar with appropriate unit and scaling.
        progress = tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name, dynamic_ncols=True)
        with open(save_dir / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    progress.update(len(chunk))
        progress.close()

if __name__ == '__main__':
    # Option 1: Use a predefined list of model filenames.
    rvc_model_names = ['hubert_base.pt', 'rmvpe.pt']
    
    # Option 2 (optional): Dynamically fetch model filenames from the page.
    # Uncomment the following two lines if you want to parse the page for .pt links.
    # print("Parsing page for available model files...")
    # rvc_model_names = get_model_links(RVC_DOWNLOAD_LINK)
    
    for model in rvc_model_names:
        print(f'Downloading {model}...')
        dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)
    
    print('All models downloaded!')
