from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from utils import determine_big_category, allowed_file, hash_image, credentials_to_dict, generateOTP
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from email.mime.multipart import MIMEMultipart
from google_auth_oauthlib.flow import Flow
from flask_sqlalchemy import SQLAlchemy
from email.mime.text import MIMEText
from model import classify_image
from dotenv import load_dotenv
from datetime import datetime
from functools import wraps
import requests
import smtplib
import uuid
import os

# Load environment variables
load_dotenv()

# Email Configuration
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTPS_SERVER = "smtp.gmail.com"
SMTPS_PORT = 465  # Port 465 for SMTPS

# Configure OAuth flow
CLIENT_SECRETS_FILE = "client_secret_961812231291-2c6sqjlrv5dfgdp5he3agtj8uj3snirp.apps.googleusercontent.com.json"  # Path to the JSON file
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Allow HTTP for local testing
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Flask App Configuration
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///inventory.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# JWT Configuration
app.config["JWT_SECRET_KEY"] = "your_jwt_secret_key"  # Change this to a strong secret in production.
jwt = JWTManager(app)

# Create a timed serializer for generating tokens
serializer = URLSafeTimedSerializer(app.secret_key)

# Create uploads directory
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize Database
db = SQLAlchemy(app)

# Flask-Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message_category = 'warning'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=True) # If the user register a new account without google authentication, username will temporarily None
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=True)   # If the user register a new account using google authentication, password will temporarily None
    reset_token = db.Column(db.String(150), nullable=True)  # For password reset
    token_expiry = db.Column(db.DateTime, nullable=True)  # Expiry time for the reset token
    
# Login History Model
class LoginHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)  # Nullable for failed attempts
    login_time = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False)  # e.g., "Success" or "Failed"

# Inventory Model
class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    image_hash = db.Column(db.String(64), nullable=False)
    big_category = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    date_added = db.Column(db.String(100), nullable=False)
    
# Edit History Model
class EditHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey("inventory.id"), nullable=False)
    previous_category = db.Column(db.String(100), nullable=False)
    new_category = db.Column(db.String(100), nullable=False)
    date_edited = db.Column(db.String(100), nullable=False)
    
# Delete History Model
class DeleteHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    image_hash = db.Column(db.String(64), nullable=False)  # SHA-256 hash
    big_category = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    date_deleted = db.Column(db.String(100), nullable=False)
    
# Manually Categorized Image Model
class ManuallyCategorizedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    image_hash = db.Column(db.String(64), unique=True, nullable=False)  # SHA-256 hash
    latest_big_category = db.Column(db.String(100), nullable=False)
    latest_category = db.Column(db.String(100), nullable=False)
    date_updated = db.Column(db.String(100), nullable=False)

# Load User for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Command to clean up orphaned uploads
def cleanup_uploads():
    """Remove files in ./static/uploads not linked to any inventory item."""
    files_in_db = {item.filename for item in Inventory.query.all()}
    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        if filename not in files_in_db:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            os.remove(file_path)
            app.logger.info(f"Deleted orphaned file: {file_path}")
            
def send_email(recipient, subject, body):
    """Send an email to the specified recipient."""
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "html"))

        # Connect using SMTPS
        with smtplib.SMTP_SSL(SMTPS_SERVER, SMTPS_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient, msg.as_string())
    except Exception as e:
        app.logger.error(f"Failed to send email: {e}")
        
def generate_reset_token(user_email):
    return serializer.dumps(user_email, salt="password-reset-salt")

def verify_reset_token(token, expiration=600):  # Token expires in 10 minutes
    try:
        email = serializer.loads(token, salt="password-reset-salt", max_age=expiration)
    except Exception:
        return None
    return email
        
def username_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.is_authenticated and not current_user.username:
            # Inform the client to set username.
            return jsonify({"success": False, "message": "Username required"}), 400 
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route("/")
@login_required
@username_required
def index():
    # Fetch user's inventory items
    items = Inventory.query.filter_by(user_id=current_user.id).all()
    items_list = []
    for item in items:
        items_list.append({
            "id": item.id,
            "filename": item.filename,
            "image_hash": item.image_hash,
            "big_category": item.big_category,
            "category": item.category,
            "date_added": item.date_added
        })
    return jsonify({"items": items_list})

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.get_json() or {}
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required."}), 400

        # Check if email already used
        if User.query.filter_by(email=email).first():
            return jsonify({"success": False, "message": "This E-mail is already used."}), 400

        # Send verification code to the email
        OTP = generateOTP()
        session["email"] = email
        session["password"] = password  # Store temporarily in the session (hashed later)
        session["otp"] = OTP

        # Send the OTP email
        try:
            send_email(
                recipient=email,
                subject="Account Registration Verification Code",
                body=f"Hi,<br><br>This is your OTP code: {OTP}.<br>If you did not request an account registration, please ignore this email."
            )
            flash(f"Verification code has been sent to your E-mail. Check your inbox.", "info")
            return redirect(url_for("verify_otp"))  # Redirect to OTP verification page
        except Exception as e:
            flash(f"Failed to send email: {str(e)}", "error")
            return redirect(url_for("register"))
    return render_template("register.html")

@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        user_otp = request.form.get("otp")
        stored_otp = session.get("otp")
        email = session.get("email")
        password = session.get("password")

        if user_otp == stored_otp:
            # OTP is correct; complete registration
            hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
            new_user = User(email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()

            # Clear the session
            session.pop("otp", None)
            session.pop("email", None)
            session.pop("password", None)

            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash("Invalid OTP. Please try again.", "error")
            return redirect(url_for("verify_otp"))

    return render_template("otp_verification.html")

@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    email = session.get("email")
    if not email:
        flash("Session expired. Please start the registration process again.", "error")
        return redirect(url_for("register"))

    OTP = generateOTP()
    session["otp"] = OTP

    try:
        send_email(
            recipient=email,
            subject="Resend: Account Registration Verification Code",
            body=f"Hi,<br><br>This is your new OTP code: {OTP}.<br><br>If you did not request an account registration, please ignore this email."
        )
        flash("OTP has been resent to your email.", "info")
    except Exception as e:
        flash(f"Failed to resend OTP: {str(e)}", "error")
    return redirect(url_for("verify_otp"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        
        # Check user credentials
        user = User.query.filter_by(email=email).first()
        login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            new_login = LoginHistory(user_id=user.id, login_time=login_time, status="Success")
            db.session.add(new_login)
            db.session.commit()
            if not user.username:
                return redirect(url_for("set_username"))
            flash("Log in successfully!", "success")
            return redirect(url_for("index"))
        
        # Log failed login (no user_id if username is invalid)
        new_login = LoginHistory(
            user_id=user.id if user else None, 
            login_time=login_time, 
            status="Failed"
        )
        db.session.add(new_login)
        db.session.commit()
        
        # Pass error to the login form
        
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

@app.route('/set-username', methods=['GET', 'POST'])
@login_required
def set_username():
    if current_user.username:  # Redirect if username already set
        return redirect(url_for("index"))

    if request.method == 'POST':
        username = request.form.get('username')
        if User.query.filter_by(username=username).first():
            flash("Username is already taken. Please choose another one.", "error")
        else:
            current_user.username = username
            db.session.commit()
            flash("Username set successfully!", "success")
            return redirect(url_for("index"))

    return render_template('set_username.html')

@app.route("/login/google")
def login_google():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
        redirect_uri=url_for("login_google_callback", _external=True),
    )
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)

@app.route("/login/google/callback")
def login_google_callback():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
        redirect_uri=url_for("login_google_callback", _external=True),
    )
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    session["credentials"] = credentials_to_dict(credentials)

    userinfo = requests.get(
        "https://www.googleapis.com/oauth2/v1/userinfo",
        headers={"Authorization": f"Bearer {credentials.token}"},
    ).json()

    email = userinfo["email"]
    username = userinfo["name"]

    # Check if user exists, otherwise create a new user
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email,
                    username=username
        )
        db.session.add(user)
        db.session.commit()

    login_user(user)
    new_login = LoginHistory(user_id=user.id, login_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status="Success")
    db.session.add(new_login)
    db.session.commit()
    return redirect(url_for("set_password"))

@app.route('/set-password', methods=['GET', 'POST'])
@login_required
@username_required
def set_password():
    if current_user.password:  # Redirect if password already set
        return redirect(url_for("index"))

    if request.method == 'POST':
        password = request.form.get('password')
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        current_user.password = hashed_password
        db.session.commit()
        flash("Password set successfully!", "success")
        return redirect(url_for("index"))

    return render_template('set_password.html')

@app.route("/logout", methods=["GET", "POST"])
@login_required
@username_required
def logout():
    if request.method == "POST":
        logout_user()
        flash("You have been logged out.", "info")
        return redirect(url_for("login"))
    
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        user = User.query.filter_by(email=email).first()
        if user:
            # Generate a reset token
            token = generate_reset_token(email)
            reset_url = url_for("reset_password", token=token, _external=True)
            
            # Send the email
            try:
                send_email(
                    recipient=email,
                    subject="Password Reset Request",
                    body=f"Hi {user.username},<br><br>To reset your password, click the following link:<br>{reset_url}<br><br>This link will expire in 1 hour.<br>If you did not request a password reset, please ignore this email."
                )
                flash("A password reset link has been sent to your email.", "info")
            except Exception as e:
                flash(f"Failed to send email: {str(e)}", "error")
        else:
            flash("No account found with that email address.", "error")
        return redirect(url_for("forgot_password"))
    
    return render_template("forgot_password.html")

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    email = verify_reset_token(token)
    if not email:
        flash("The password reset link is invalid or has expired.", "error")
        return redirect(url_for("forgot_password"))
    
    if request.method == "POST":
        new_password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(new_password, method="pbkdf2:sha256")
            db.session.commit()
            flash("Your password has been reset. You can now log in.", "success")
            return redirect(url_for("login"))
        else:
            flash("User not found.", "error")
    
    return render_template("reset_password.html", token=token)

@app.route("/delete-account", methods=["POST"])
@login_required
@username_required
def delete_account():
    try:
        user_id = current_user.id

        # Remove user files
        user_items = Inventory.query.filter_by(user_id=user_id).all()
        for item in user_items:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], item.filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Delete related user data
        User.query.filter_by(id=user_id).delete()
        LoginHistory.query.filter_by(user_id=user_id).delete()
        Inventory.query.filter_by(user_id=user_id).delete()
        EditHistory.query.filter(EditHistory.item_id.in_([item.id for item in user_items])).delete()
        DeleteHistory.query.filter_by(user_id=user_id).delete()
        ManuallyCategorizedImage.query.filter_by(user_id=user_id).delete()

        # Delete user record
        db.session.delete(current_user)
        db.session.commit()

        # Log out the user after deletion
        logout_user()
        flash("Your account has been deleted successfully.", "info")
        return redirect(url_for("login"))

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting account: {e}")
        flash("An error occurred while deleting your account. Please try again later.", "error")
        return redirect(url_for("index"))


@app.route("/upload", methods=["POST"])
@login_required
@username_required
def upload():    
    files = request.files.getlist("images")
    upload = False

    for file in files:
        if file:
            if not allowed_file(file.filename):
                flash(f"Invalid file type: {os.path.splitext(file.filename)[1]}", "error")
                continue
            
            # Generate a unique filename
            original_filename = file.filename
            unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(filepath)
            image_hash = hash_image(filepath)
            
            # Avoid duplicates
            existed_item = Inventory.query.filter_by(user_id=current_user.id, image_hash=image_hash).first()
            if existed_item:
                flash(f"Item is already exist. Date added: {existed_item.date_added}", "warning")
                continue
            
            # Check for manually categorized history
            categorized_item = ManuallyCategorizedImage.query.filter_by(image_hash=image_hash).first()
                     
            # Preprocess and classify the image
            try:
                date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if categorized_item:
                    big_category = categorized_item.latest_big_category
                    category = categorized_item.latest_category
                else:
                    big_category, category = classify_image(filepath)
                    
                # Add to the database
                new_item = Inventory(
                    user_id=current_user.id, 
                    filename=unique_filename, 
                    image_hash=image_hash,
                    big_category=big_category, 
                    category=category, 
                    date_added=date_added
                )
                db.session.add(new_item)
                upload = True
            except Exception as e:
                flash(f"Failed to classify image {file.filename}: {e}", "error")
        else:
            flash ("No images uploaded.", "warning")
            return redirect(url_for("index"))
    if upload:
        flash(f"Item(s) uploaded successfully.", "success")
    db.session.commit()
    
    return redirect(url_for("index"))

@app.route("/delete", methods=["POST"])
@login_required
@username_required
def delete_items():
    # Get selected item IDs from the form
    item_ids = request.form.getlist("item_ids")
    
    items_to_delete = Inventory.query.filter(Inventory.id.in_(item_ids), Inventory.user_id == current_user.id).all()
    
    if not items_to_delete:
        flash("No valid items found to delete.", "warning")
        return redirect(url_for("index"))
    
    # Iterate over the items and delete associated files
    for item in items_to_delete:
        file_path = os.path.join("./static/uploads", item.filename)
        image_hash = hash_image(file_path)
        existing_deleted_file = DeleteHistory.query.filter(DeleteHistory.image_hash == image_hash).first()
        
        # Log delete history
        if existing_deleted_file:
            existing_deleted_file.big_category = item.big_category
            existing_deleted_file.category = item.category
            existing_deleted_file.date_deleted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            delete_history = DeleteHistory(
                user_id=current_user.id,
                image_hash=image_hash,
                big_category=item.big_category,
                category=item.category,
                date_deleted=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            db.session.add(delete_history)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                app.logger.error(f"Error deleting file {item.filename}: {e}")
                flash(f"Error deleting file {item.filename}: {e}", "error")
        else:
            flash(f"File {item.filename} not found.", "warning")
    
    # Delete items from the inventory
    Inventory.query.filter(Inventory.id.in_(item_ids), Inventory.user_id == current_user.id).delete(synchronize_session=False)
    EditHistory.query.filter(EditHistory.item_id.in_(item_ids)).delete(synchronize_session=False)
    db.session.commit()
    
    # Clean up orphaned uploads
    cleanup_uploads()

    flash("Selected items have been deleted.", "success")
    return redirect(url_for("index"))

@app.route("/edit", methods=["POST"])
@login_required
@username_required
def edit_item():
    # Get the item ID and new category from the form
    data = request.get_json()
    item_id = data.get("item_id")
    new_category = data.get("category")
    new_big_category = determine_big_category(new_category)
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Find the item in the database
    item = Inventory.query.filter_by(id=item_id, user_id=current_user.id).first()
    
    if item:
        image_hash = hash_image(os.path.join(app.config["UPLOAD_FOLDER"], item.filename))
        # Track editing history
        edit_history = EditHistory(
            item_id=item.id,
            previous_category=item.category,
            new_category=new_category,
            date_edited=date_time
        )
        db.session.add(edit_history)
        
        manually_categorized = ManuallyCategorizedImage.query.filter_by(image_hash=image_hash).first()
        if manually_categorized:
            manually_categorized.user_id = current_user.id
            manually_categorized.latest_big_category = new_big_category
            manually_categorized.latest_category = new_category
            manually_categorized.date_update = date_time
        else:
            manually_categorized = ManuallyCategorizedImage(
                user_id=current_user.id,
                image_hash=image_hash,
                latest_big_category=new_big_category,
                latest_category=new_category,
                date_updated=date_time
            )
            db.session.add(manually_categorized)
        
        # Update category and big_category
        item.category = new_category
        item.big_category = new_big_category
        db.session.commit()

        return jsonify({"success": True, "message": "Item updated successfully."})
    else:
        return jsonify({"success": False, "message": "Item not found or unauthorized access."}), 404

# Initialize Database
with app.app_context():
    db.create_all()

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, ssl_context=("cert.pem", "key.pem"))