import os
import requests
import time
import urllib.parse

def search_wikimedia_images(query, max_results=10):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6", # File namespace
        "gsrlimit": max_results,
        "iiprop": "url|size"
    }
    headers = {
        'User-Agent': 'AeroSentinel/1.0 (Contact: aerosentinel@example.com)'
    }
    
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    
    image_urls = []
    if 'query' in data and 'pages' in data['query']:
        for page_id, page_info in data['query']['pages'].items():
            if 'imageinfo' in page_info:
                info = page_info['imageinfo'][0]
                url = info.get('url', '')
                if url.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_urls.append(url)
                    
    return image_urls

def main():
    target_dir = r"."\aircraft_backgrounds"
    os.makedirs(target_dir, exist_ok=True)
    
    queries = [
        "aircraft fuselage skin",
        "aircraft rivets texture",
        "unpainted aircraft aluminum",
        "aircraft panel close up",
        "airplane skin metal"
    ]
    
    all_urls = set()
    for q in queries:
        print(f"Searching for: {q}")
        urls = search_wikimedia_images(q, max_results=10)
        all_urls.update(urls)
        time.sleep(1) # Be polite to API
        
    print(f"Found {len(all_urls)} unique candidate images.")
    
    # Download them
    downloaded = 0
    for i, url in enumerate(all_urls):
        if downloaded >= 15:
            break
        try:
            print(f"Downloading {i+1}/{len(all_urls)}: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://commons.wikimedia.org/'
            }
            img_resp = requests.get(url, headers=headers, stream=True, timeout=10)
            
            if img_resp.status_code == 200:
                # Save it
                ext = url.split('.')[-1].lower()
                if ext not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpg'
                filepath = os.path.join(target_dir, f"bg_{downloaded:03d}.{ext}")
                with open(filepath, 'wb') as f:
                    for chunk in img_resp.iter_content(1024):
                        f.write(chunk)
                downloaded += 1
            else:
                print(f"Failed HTTP {img_resp.status_code}")
            time.sleep(1)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            
    print(f"Successfully downloaded {downloaded} images to {target_dir}")

if __name__ == "__main__":
    main()
