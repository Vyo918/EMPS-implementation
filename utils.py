import hashlib
import random
import math

# Helper function to determine Big_Category
def determine_big_category(category):
    if category == "Dress":
        return "Dress"
    elif category in ["Shirt", "T-shirt", "Hoodie", "Sweater", "Polo Shirt"]:
        return "Top"
    elif category in ["Pants", "Shorts", "Skirt"]:
        return "Bottom"
    elif category in ["Flats", "Heels", "Shoes", "Sneakers"]:
        return "Footwear"
    return "Unknown"

# Helper function to validate file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_image(file_path):
    """Compute the SHA-256 hash of an image file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def credentials_to_dict(credentials):
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }

def generateOTP() :
    string = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    OTP = ""
    length = len(string)
    for i in range(6) :
        OTP += string[math.floor(random.random() * length)]
    return OTP