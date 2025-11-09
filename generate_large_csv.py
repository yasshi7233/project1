"""
Generate a large CSV file (16MB) for testing purposes
"""

import pandas as pd
import numpy as np
import random
import string
import os

def generate_random_text(length=100):
    """Generate random text for catalog content"""
    words = [
        'Premium', 'Quality', 'Professional', 'Advanced', 'Digital', 'Wireless',
        'Bluetooth', 'Smart', 'High', 'Performance', 'Ultra', 'Pro', 'Max',
        'Mini', 'Compact', 'Portable', 'Durable', 'Lightweight', 'Stainless',
        'Steel', 'Plastic', 'Metal', 'Wooden', 'Ceramic', 'Glass', 'Fabric',
        'Leather', 'Rubber', 'Silicon', 'Carbon', 'Fiber', 'Aluminum', 'Copper',
        'Gold', 'Silver', 'Bronze', 'Titanium', 'Chrome', 'Matte', 'Glossy',
        'Waterproof', 'Shockproof', 'Dustproof', 'Anti-bacterial', 'Eco-friendly',
        'Recyclable', 'Biodegradable', 'Organic', 'Natural', 'Synthetic'
    ]
    
    text = ' '.join(random.choices(words, k=random.randint(10, 20)))
    return text

def generate_catalog_content():
    """Generate realistic catalog content"""
    title = generate_random_text(50)
    description = generate_random_text(100)
    quantity = random.choice([
        f"Pack of {random.randint(1, 12)}",
        f"Quantity: {random.randint(1, 50)} pieces",
        f"{random.randint(1, 10)} units per pack",
        f"Set of {random.randint(2, 20)} items"
    ])
    
    return f"{title}\n{description}\n{quantity}"

def generate_image_url():
    """Generate realistic image URLs"""
    base_urls = [
        "https://m.media-amazon.com/images/I/",
        "https://images.example.com/products/",
        "https://cdn.shopify.com/s/files/",
        "https://static.example.com/images/"
    ]
    
    base = random.choice(base_urls)
    image_id = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    return f"{base}{image_id}.jpg"

def generate_large_csv(target_size_mb=16):
    """Generate a CSV file of specified size"""
    print(f"Generating CSV file of {target_size_mb}MB...")
    
    # Estimate rows needed (rough calculation)
    # Each row is approximately 200-300 bytes
    estimated_rows = (target_size_mb * 1024 * 1024) // 250
    
    print(f"Estimated rows needed: {estimated_rows:,}")
    
    data = []
    batch_size = 10000
    
    for i in range(0, estimated_rows, batch_size):
        batch_data = []
        current_batch_size = min(batch_size, estimated_rows - i)
        
        for j in range(current_batch_size):
            sample_id = f"test_{i + j + 1:06d}"
            catalog_content = generate_catalog_content()
            image_link = generate_image_url()
            
            batch_data.append({
                'sample_id': sample_id,
                'catalog_content': catalog_content,
                'image_link': image_link
            })
        
        data.extend(batch_data)
        
        if (i + batch_size) % 50000 == 0:
            print(f"Generated {i + batch_size:,} rows...")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = f"large_test_data_{target_size_mb}mb.csv"
    df.to_csv(filename, index=False)
    
    # Check file size
    file_size = os.path.getsize(filename) / (1024 * 1024)
    
    print(f"\n[SUCCESS] CSV file generated successfully!")
    print(f"[FILE] Filename: {filename}")
    print(f"[ROWS] Rows: {len(df):,}")
    print(f"[SIZE] File size: {file_size:.2f} MB")
    print(f"[COLS] Columns: {list(df.columns)}")
    
    # Show sample data
    print(f"\n[SAMPLE] Sample data:")
    print(df.head(3).to_string())
    
    return filename

if __name__ == "__main__":
    # Generate 16MB CSV file
    filename = generate_large_csv(16)
    
    print(f"\n[READY] File ready for testing!")
    print(f"[TIP] You can use this file to test your Flask app or pipeline")
