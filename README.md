🛒 Smart Shopping Cart App


A smart, health-aware shopping cart web application designed to enhance the grocery experience. This app enables users to log in, scan products, manage a shopping cart, and receive real-time alerts for allergens based on a personalized allergy profile stored in an SQL database.

🚀 Features
🔐 User Authentication – Register and log in securely to personalize your shopping experience

📦 Cart Management – Add, update, or remove products in a smooth, intuitive interface

📷 Barcode Scanner – Scan product barcodes to auto-populate product info

⚠️ Allergy Detection System – Real-time warnings when scanned products contain allergens from the user's profile

🧠 Smart Matching – Allergy checks performed via efficient SQL queries

🎯 Clean UI – Python-powered frontend styled with CSS for a smooth and simple user experience

🧑‍💻 Tech Stack
Frontend: Python (Flask / Django / TKinter – specify if applicable), HTML, CSS

Backend: Python

Database: SQL (MySQL / PostgreSQL / SQLite – specify your DBMS)

Scanner Integration: Python libraries (e.g., OpenCV / pyzbar / etc.)

📸 Screenshots

![WhatsApp Image 2025-05-23 at 22 38 13_e2f75f98](https://github.com/user-attachments/assets/53f14520-c75a-487a-8ac3-aa3f7c423976)
![WhatsApp Image 2025-05-23 at 22 40 57_b391b628](https://github.com/user-attachments/assets/de69d67b-8870-4854-8ab0-68ed6d0e0b70)
![WhatsApp Image 2025-05-23 at 22 41 14_2d1aa810](https://github.com/user-attachments/assets/7f2c6696-5c8e-49a4-9c5e-a60fbbc6e134)

📦 Getting Started
Prerequisites
Python 3.x

pip

SQL database (MySQL / PostgreSQL / SQLite)

Barcode scanner or webcam support (for scanner feature)

Installation
Clone the repository:
git clone https://github.com/yourusername/smart-shopping-cart.git
cd smart-shopping-cart

Create and activate a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Set up your database:
Update DB credentials in config.py
Run schema setup scripts (e.g., setup.sql)

Run the app:
python app.py


🛠 Configuration
Customize allergy detection logic in allergy_checker.py

Product scanner logic in scanner.py (uses webcam or image input)

Update SQL queries in db_manager.py for optimization or extensions

🤝 Contributing
Contributions are welcome!
If you'd like to add features, fix bugs, or improve documentation, feel free to fork the repo and create a pull request.

📄 License
This project is licensed under the MIT License. See LICENSE for details.

📬 Contact
Created by Aman Chauhan & Nehal Saraswat
