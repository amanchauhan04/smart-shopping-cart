# Smart Shopping Cart Application
# -----------------------------------
# A software implementation of an RFID-based AI shopping cart system
# Features:
# - Automatic product scanning via barcode/image simulation
# - Real-time budget tracking
# - Allergen detection
# - Smart recommendations
# - Budget-aware alternatives

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageTk
import cv2
import json
import time
import re
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QTableWidget, QTableWidgetItem, QLineEdit, 
                            QComboBox, QTabWidget, QDialog, QMessageBox, QFrame, QStackedWidget,
                            QScrollArea, QGridLayout, QListWidget, QListWidgetItem, QCheckBox,
                            QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog, QSplashScreen,
                            QInputDialog)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, QThread, QUrl, QRect
from PyQt6.QtGui import QPixmap, QFont, QIcon, QColor, QPalette, QImage, QLinearGradient, QPainter
import hashlib
import random

print("Python version:", sys.version)
try:
    import PyQt6
    print("PyQt6 version:", PyQt6.QtCore.PYQT_VERSION_STR)
    print("Qt version:", PyQt6.QtCore.QT_VERSION_STR)
except Exception as e:
    print("Error importing PyQt6:", e)
print("Script completed")

# ================ CONFIGURATION ================

# Database Path
DB_PATH = 'shop_app.db'

# Color Scheme
PRIMARY_COLOR = "#4a69bd"
SECONDARY_COLOR = "#6a89cc"
ACCENT_COLOR = "#f6b93b"
TEXT_COLOR = "#2c3e50"
BACKGROUND_COLOR = "#f5f6fa"
SUCCESS_COLOR = "#78e08f"
WARNING_COLOR = "#fa983a"
DANGER_COLOR = "#eb2f06"

# ================ DATABASE SETUP ================


def create_database_tables():
    """Initialize the SQLite database with new columns matching CSV"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Drop old products table if exists
        cursor.execute("DROP TABLE IF EXISTS products")
        
        # Create new products table matching CSV columns
        cursor.execute(f"""
            CREATE TABLE products (
                
    product_id TEXT PRIMARY KEY,
    Descrip TEXT,
    Price REAL,
    Allergens TEXT,
    Energy_kcal REAL,
    Protein_g REAL,
    Saturated_fats_g REAL,
    Fat_g REAL,
    Carb_g REAL,
    Fiber_g REAL,
    Sugar_g REAL,
    Calcium_mg REAL,
    Iron_mg REAL,
    Magnesium_mg REAL,
    Phosphorus_mg REAL,
    Potassium_mg REAL,
    Sodium_mg REAL,
    Zinc_mg REAL,
    Copper_mcg REAL,
    Manganese_mg REAL,
    Selenium_mcg REAL,
    VitC_mg REAL,
    Thiamin_mg REAL,
    Riboflavin_mg REAL,
    Niacin_mg REAL,
    VitB6_mg REAL,
    Folate_mcg REAL,
    VitB12_mcg REAL,
    VitA_mcg REAL,
    VitE_mg REAL,
    VitD2_mcg REAL

            )
        """)
        
        print("Database tables created successfully with updated columns")
        conn.commit()
        
    except sqlite3.Error as err:
        print(f"Error: {err}")
    
    finally:
        if conn:
            cursor.close()
            conn.close()


    """Initialize the SQLite database with required tables if they don't exist"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                userDescrip TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                first_Descrip TEXT,
                last_Descrip TEXT,
                budget REAL DEFAULT 0.00,
                creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
                
        # Create Allergens table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS allergens (
                allergen_id INTEGER PRIMARY KEY AUTOINCREMENT,
                Descrip TEXT UNIQUE NOT NULL
            )
        """)
        
        # Create Product_Allergens table (Many-to-Many relationship)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS product_allergens (
                product_id TEXT,
                allergen_id INTEGER,
                PRIMARY KEY (product_id, allergen_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
                FOREIGN KEY (allergen_id) REFERENCES allergens(allergen_id) ON DELETE CASCADE
            )
        """)
        
        # Create User_Allergens table (Many-to-Many relationship)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_allergens (
                user_id INTEGER,
                allergen_id INTEGER,
                PRIMARY KEY (user_id, allergen_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (allergen_id) REFERENCES allergens(allergen_id) ON DELETE CASCADE
            )
        """)
        
        # Create Cart table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cart (
                cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Create Cart_Items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cart_items (
                cart_id INTEGER,
                product_id TEXT,
                quantity INTEGER DEFAULT 1,
                PRIMARY KEY (cart_id, product_id),
                FOREIGN KEY (cart_id) REFERENCES cart(cart_id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)
        
        # Create Purchase_History table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS purchase_history (
                purchase_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_amount REAL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Create Purchase_Items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS purchase_items (
                purchase_id INTEGER,
                product_id TEXT,
                quantity INTEGER,
                price_at_purchase REAL,
                PRIMARY KEY (purchase_id, product_id),
                FOREIGN KEY (purchase_id) REFERENCES purchase_history(purchase_id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)
        
        print("Database tables created successfully")
        conn.commit()
        
    except sqlite3.Error as err:
        print(f"Error: {err}")
    
    finally:
        if conn:
            cursor.close()
            conn.close()


def import_dataset(dataset_path):
    """Import dataset from CSV with all columns mapped directly"""
    try:
        df = pd.read_csv(dataset_path)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for _, row in df.iterrows():
            product_id = str(row['NDB_No'])  # Use NDB_No as primary key

            # Insert all columns
            cursor.execute("""
                INSERT OR REPLACE INTO products (
                    product_id, Descrip, Price, Allergens, Energy_kcal, Protein_g,
                    Saturated_fats_g, Fat_g, Carb_g, Fiber_g, Sugar_g, Calcium_mg,
                    Iron_mg, Magnesium_mg, Phosphorus_mg, Potassium_mg, Sodium_mg,
                    Zinc_mg, Copper_mcg, Manganese_mg, Selenium_mcg, VitC_mg,
                    Thiamin_mg, Riboflavin_mg, Niacin_mg, VitB6_mg, Folate_mcg,
                    VitB12_mcg, VitA_mcg, VitE_mg, VitD2_mcg
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product_id,
                row['Descrip'],
                row['Price'],
                row['Allergens'],
                row['Energy_kcal'],
                row['Protein_g'],
                row['Saturated_fats_g'],
                row['Fat_g'],
                row['Carb_g'],
                row['Fiber_g'],
                row['Sugar_g'],
                row['Calcium_mg'],
                row['Iron_mg'],
                row['Magnesium_mg'],
                row['Phosphorus_mg'],
                row['Potassium_mg'],
                row['Sodium_mg'],
                row['Zinc_mg'],
                row['Copper_mcg'],
                row['Manganese_mg'],
                row['Selenium_mcg'],
                row['VitC_mg'],
                row['Thiamin_mg'],
                row['Riboflavin_mg'],
                row['Niacin_mg'],
                row['VitB6_mg'],
                row['Folate_mcg'],
                row['VitB12_mcg'],
                row['VitA_mcg'],
                row['VitE_mg'],
                row['VitD2_mcg']
            ))

        conn.commit()
        print(f"Successfully imported {{len(df)}} products from {{dataset_path}}")

    except Exception as e:
        print(f"Error importing dataset: {{e}}")

    finally:
        if conn:
            cursor.close()
            conn.close()


    """Import dataset from CSV file to SQLite database"""
    try:
        # Read CSV file
        df = pd.read_csv(dataset_path)
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Extract potential allergens from nutritional values
        allergens = set()
        
        # Insert products
        for _, row in df.iterrows():
            product_id = row['ProductID']
            Descrip = row['Name']
            category = row['Category']
            product_type = row['Type']
            brand = row['Brand']
            price = float(row['Price'])
            quantity = row['Quantity']
            nutritional_values = row['Nutritional Values']
            
            # Check if product already exists
            cursor.execute("SELECT product_id FROM products WHERE product_id = ?", (product_id,))
            if cursor.fetchone() is not None:
                # Update existing product
                cursor.execute("""
                    UPDATE products 
                    SET Descrip = ?, category = ?, type = ?, brand = ?, 
                        price = ?, quantity = ?, nutritional_values = ?
                    WHERE product_id = ?
                """, (Descrip, category, product_type, brand, price, quantity, nutritional_values, product_id))
            else:
                # Insert new product
                cursor.execute("""
                    INSERT INTO products 
                    (product_id, Descrip, category, type, brand, price, quantity, nutritional_values)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (product_id, Descrip, category, product_type, brand, price, quantity, nutritional_values))
            
            # Extract potential allergens from nutritional values
            if nutritional_values and nutritional_values != 'N/A':
                try:
                    # Extract nutritional info using regex
                    nutrition_dict = {}
                    matches = re.findall(r"'([^']+)':\s*([0-9.]+)", nutritional_values)
                    for key, value in matches:
                        nutrition_dict[key] = float(value)
                    
                    # Common allergens to check for in food items
                    if category == 'Groceries':
                        # Check common allergens based on product type
                        if product_type == 'Milk' or 'Milk' in Descrip:
                            allergens.add('lactose')
                        elif product_type == 'Cereal':
                            allergens.add('gluten')
                        elif 'Nuts' in Descrip or 'nut' in Descrip.lower():
                            allergens.add('nuts')
                except Exception as e:
                    print(f"Error processing nutritional values for {product_id}: {e}")
        
        # Insert allergens
        for allergen in allergens:
            try:
                cursor.execute("INSERT OR IGNORE INTO allergens (Descrip) VALUES (?)", (allergen,))
            except sqlite3.IntegrityError:
                pass  # Allergen already exists
        
        # Commit changes
        conn.commit()
        print(f"Successfully imported {len(df)} products from {dataset_path}")
        
    except Exception as e:
        print(f"Error importing dataset: {e}")
        
    finally:
        if conn:
            cursor.close()
            conn.close()

# ================ RECOMMENDATION ENGINE ================

class ProductEmbedding(nn.Module):
    def __init__(self, num_products, embedding_dim=32):
        super(ProductEmbedding, self).__init__()
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
    
    def forward(self, product_idx):
        return self.product_embedding(product_idx)

class RecommendationModel(nn.Module):
    def __init__(self, num_products, embedding_dim=32, hidden_dim=64):
        super(RecommendationModel, self).__init__()
        
        # Product embeddings
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        
        # Neural network layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_products)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, product_indices):
        """
        Forward pass for getting recommendations
        
        Args:
            product_indices: Tensor of shape (batch_size, num_items_in_cart)
        
        Returns:
            Scores for all products
        """
        # Get embeddings for products in cart
        embedded = self.product_embedding(product_indices)
        
        # Average the embeddings of products in the cart
        cart_embedding = torch.mean(embedded, dim=1)
        
        # Pass through neural network
        x = self.relu(self.fc1(cart_embedding))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class RecommendationEngine:
    def __init__(self):
        self.model = None
        self.product_to_idx = {}
        self.idx_to_product = {}
        self.category_rules = {
            'Bread': ['Butter', 'Jam'],
            'Cereal': ['Milk'],
            'Coffee': ['Milk', 'Sugar'],
            'Pasta': ['Sauce'],
            'Shoes': ['Socks'],
            'Smartphone': ['Charger', 'Headphones'],
            'Laptop': ['Mouse', 'Keyboard']
        }
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize recommendation model"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get all products
            cursor.execute("SELECT product_id, Descrip, category, type FROM products")
            products = cursor.fetchall()
            
            # Create mappings
            for i, (product_id, _, _, _) in enumerate(products):
                self.product_to_idx[product_id] = i
                self.idx_to_product[i] = product_id
            
            # Initialize model
            num_products = len(products)
            self.model = RecommendationModel(num_products)
            
            # Mock training (in a real implementation, we'd train on historical data)
            # Here we're just initializing with random weights
            print(f"Initialized recommendation model with {num_products} products")
            
            # Create type-based relationships for rule-based recommendations
            self.type_relationships = {}
            for prod_id, Descrip, category, prod_type in products:
                if prod_type not in self.type_relationships:
                    self.type_relationships[prod_type] = []
                self.type_relationships[prod_type].append(prod_id)
                
        except Exception as e:
            print(f"Error initializing recommendation model: {e}")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def get_recommendations(self, cart_items, user_id=None, budget=None):
        """
        Get product recommendations
        
        Args:
            cart_items: List of product IDs in the cart
            user_id: Optional user ID for personalized recommendations
            budget: Optional budget constraint
            
        Returns:
            List of recommended product IDs
        """
        try:
            recommendations = []
            
            # If cart is empty, return popular items
            if not cart_items:
                return self.get_popular_items(5)
            
            # 1. Rule-based recommendations
            rule_based = self.get_rule_based_recommendations(cart_items)
            recommendations.extend(rule_based[:2])  # Add top 2 rule-based recommendations
            
            # 2. Model-based recommendations
            model_based = self.get_model_based_recommendations(cart_items)
            recommendations.extend(model_based[:3])  # Add top 3 model-based recommendations
            
            # Filter out items already in cart
            recommendations = [r for r in recommendations if r not in cart_items]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for item in recommendations:
                if item not in seen:
                    seen.add(item)
                    unique_recommendations.append(item)
            
            # Apply budget constraint if provided
            if budget is not None:
                unique_recommendations = self.filter_by_budget(unique_recommendations, budget, cart_items)
            
            return unique_recommendations[:5]  # Return top 5 unique recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def get_rule_based_recommendations(self, cart_items):
        """Get recommendations based on predefined rules"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            recommendations = []
            
            # For each item in cart, find complementary products
            for product_id in cart_items:
                cursor.execute("""
                    SELECT type, Descrip FROM products WHERE product_id = ?
                """, (product_id,))
                result = cursor.fetchone()
                
                if result:
                    product_type, product_Descrip = result
                    
                    # Check type-based relationships
                    if product_type in self.type_relationships:
                        related_products = self.type_relationships[product_type]
                        related_products = [p for p in related_products if p != product_id]
                        recommendations.extend(related_products)
                    
                    # Check Descrip-based rules
                    for keyword, complements in self.category_rules.items():
                        if keyword.lower() in product_Descrip.lower():
                            for complement in complements:
                                cursor.execute("""
                                    SELECT product_id FROM products 
                                    WHERE Descrip LIKE ? OR type LIKE ?
                                    LIMIT 2
                                """, (f"%{complement}%", f"%{complement}%"))
                                results = cursor.fetchall()
                                if results:
                                    recommendations.extend([r[0] for r in results])
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting rule-based recommendations: {e}")
            return []
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def get_model_based_recommendations(self, cart_items):
        """Get recommendations from the neural network model"""
        try:
            # Convert product IDs to indices
            cart_indices = [self.product_to_idx.get(pid) for pid in cart_items if pid in self.product_to_idx]
            
            if not cart_indices:
                return []
            
            # Create input tensor
            cart_tensor = torch.LongTensor(cart_indices)
            
            # Get recommendations from model
            with torch.no_grad():
                # In a real implementation, we'd use the model here
                # For demo purposes, just return random products
                num_products = len(self.product_to_idx)
                random_indices = torch.randperm(num_products)[:10]
                
                # Convert indices back to product IDs
                recommendations = [self.idx_to_product[idx.item()] 
                                  for idx in random_indices 
                                  if idx.item() in self.idx_to_product]
                
                return recommendations
                
        except Exception as e:
            print(f"Error getting model-based recommendations: {e}")
            return []
    
    def get_popular_items(self, count=5):
        """Get popular items as fallback recommendations"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT product_id FROM products
                ORDER BY RANDOM()
                LIMIT ?
            """, (count,))
            
            popular_items = [row[0] for row in cursor.fetchall()]
            return popular_items
            
        except Exception as e:
            print(f"Error getting popular items: {e}")
            return []
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def filter_by_budget(self, recommendations, budget, cart_items):
        """Filter recommendations to stay within budget"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Calculate current cart total
            placeholders = ','.join(['?' for _ in cart_items])
            cursor.execute(f"""
                SELECT SUM(price) FROM products 
                WHERE product_id IN ({placeholders})
            """, cart_items)
            
            result = cursor.fetchone()
            current_total = float(result[0]) if result[0] else 0
            
            # Calculate remaining budget
            remaining_budget = budget - current_total
            
            if remaining_budget <= 0:
                return []  # No budget left
            
            # Filter recommendations by price
            filtered = []
            for product_id in recommendations:
                cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
                result = cursor.fetchone()
                if result and float(result[0]) <= remaining_budget:
                    filtered.append(product_id)
            
            return filtered
            
        except Exception as e:
            print(f"Error filtering by budget: {e}")
            return recommendations
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def get_budget_alternatives(self, product_id, budget_remaining):
        """Get cheaper alternative for a product"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get product details
            cursor.execute("""
                SELECT category, type, price FROM products 
                WHERE product_id = ?
            """, (product_id,))
            
            result = cursor.fetchone()
            if not result:
                return []
                
            category, product_type, price = result
            
            # Find cheaper alternatives in the same category and type
            cursor.execute("""
                SELECT product_id FROM products 
                WHERE category = ? AND type = ? AND price < ? AND price <= ?
                ORDER BY price DESC
                LIMIT 3
            """, (category, product_type, price, budget_remaining))
            
            alternatives = [row[0] for row in cursor.fetchall()]
            return alternatives
            
        except Exception as e:
            print(f"Error getting budget alternatives: {e}")
            return []
            
        finally:
            if conn:
                cursor.close()
                conn.close()

# ================ USER INTERFACE ================

class ScannerSimulator(QWidget):
    """Widget to simulate the RFID/barcode scanner"""
    scanComplete = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scanner Simulator")
        self.setGeometry(100, 100, 600, 500)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Camera viewfinder
        camera_layout = QVBoxLayout()
        
        # Camera view placeholder
        self.camera_view = QLabel("Camera Simulation")
        self.camera_view.setStyleSheet(f"""
            background-color: #000;
            color: white;
            font-size: 18px;
            min-height: 300px;
            border-radius: 8px;
            border: 2px solid {PRIMARY_COLOR};
        """)
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_layout.addWidget(self.camera_view)
        
        # Camera controls
        controls_layout = QHBoxLayout()
        
        self.scan_button = QPushButton("Scan Product")
        self.scan_button.setStyleSheet(f"""
            background-color: {PRIMARY_COLOR};
            color: white;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 4px;
        """)
        self.scan_button.clicked.connect(self.scan_product)
        controls_layout.addWidget(self.scan_button)
        
        self.manual_button = QPushButton("Manual Entry")
        self.manual_button.setStyleSheet(f"""
            background-color: {SECONDARY_COLOR};
            color: white;
            padding: 12px;
            font-size: 16px;
            border-radius: 4px;
        """)
        self.manual_button.clicked.connect(self.manual_entry)
        controls_layout.addWidget(self.manual_button)
        
        camera_layout.addLayout(controls_layout)
        layout.addLayout(camera_layout)
        
        # Product display section
        product_layout = QHBoxLayout()
        
        self.product_image = QLabel()
        self.product_image.setFixedSize(150, 150)
        self.product_image.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            padding: 5px;
        """)
        self.product_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        product_layout.addWidget(self.product_image)
        
        product_info_layout = QVBoxLayout()
        
        self.product_Descrip = QLabel("Scan a product...")
        self.product_Descrip.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)
        product_info_layout.addWidget(self.product_Descrip)
        
        self.product_price = QLabel("")
        self.product_price.setStyleSheet("""
            font-size: 16px;
            color: #555;
        """)
        product_info_layout.addWidget(self.product_price)
        
        self.product_brand = QLabel("")
        product_info_layout.addWidget(self.product_brand)
        
        self.add_button = QPushButton("Add to Cart")
        self.add_button.setStyleSheet(f"""
            background-color: {SUCCESS_COLOR};
            color: white;
            padding: 10px;
            font-weight: bold;
            border-radius: 4px;
        """)
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(self.add_to_cart)
        product_info_layout.addWidget(self.add_button)
        
        product_layout.addLayout(product_info_layout)
        layout.addLayout(product_layout)
        
        self.setLayout(layout)
        
        # Current scanned product
        self.current_product_id = None
        
        # Timer to simulate scanning
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.update_scan_animation)
        self.scan_frame = 0
        
        # Create a directory for product images if it doesn't exist
        if not os.path.exists("images"):
            os.makedirs("images")
    
    def scan_product(self):
        """Simulate scanning a product"""
        self.scan_button.setEnabled(False)
        self.scan_button.setText("Scanning...")
        self.product_Descrip.setText("Scanning...")
        self.product_price.setText("")
        self.product_brand.setText("")
        self.add_button.setEnabled(False)
        
        # Simulate scanner animation
        self.scan_frame = 0
        self.scan_timer.start(100)  # Update every 100ms
    
    def update_scan_animation(self):
        """Update the scanning animation"""
        scan_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.scan_frame = (self.scan_frame + 1) % len(scan_frames)
        self.product_Descrip.setText(f"Scanning... {scan_frames[self.scan_frame]}")
        
        # After 2 seconds (20 frames), complete scan
        if self.scan_frame == 0 and self.scan_timer.isActive():
            self.scan_timer.stop()
            self.complete_scan()
    
    def complete_scan(self):
        """Complete the scanning process"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get random product
            cursor.execute("""
                SELECT product_id, Descrip, brand, price, category, type 
                FROM products 
                ORDER BY RANDOM() 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            if result:
                product_id, Descrip, brand, price, category, product_type = result
                
                self.current_product_id = product_id
                self.product_Descrip.setText(Descrip)
                self.product_price.setText(f"₹ {price:.2f}")
                self.product_brand.setText(f"{brand} - {category} / {product_type}")
                self.add_button.setEnabled(True)
                
                # Set product image (placeholder or random image)
                image_path = f"images/{product_id}.jpg"
                if not os.path.exists(image_path):
                    # Generate a placeholder color based on product category
                    color_map = {
                        'Groceries': '#78e08f',
                        'Beauty': '#a29bfe',
                        'Electronics': '#74b9ff',
                        'Clothing': '#ff7675',
                        'Furniture': '#fdcb6e'
                    }
                    color = color_map.get(category, '#dfe6e9')
                    
                    # Create placeholder image
                    img = QPixmap(150, 150)
                    img.fill(QColor(color))
                    painter = QPainter(img)
                    painter.setPen(Qt.GlobalColor.white)
                    painter.setFont(QFont('Arial', 12, QFont.Weight.Bold))
                    painter.drawText(QRect(0, 0, 150, 150), Qt.AlignmentFlag.AlignCenter, product_type[0].upper())
                    painter.end()
                    
                    self.product_image.setPixmap(img)
                else:
                    self.product_image.setPixmap(QPixmap(image_path).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio))
                
                self.scan_button.setText("Scan Another")
                self.scan_button.setEnabled(True)
                
        except Exception as e:
            print(f"Error completing scan: {e}")
            self.product_Descrip.setText("Scan error. Try again.")
            self.scan_button.setText("Scan Product")
            self.scan_button.setEnabled(True)
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def manual_entry(self):
        """Show dialog for manual product selection"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get product list
            cursor.execute("SELECT product_id, Descrip, brand, price FROM products ORDER BY Descrip")
            products = cursor.fetchall()
            
            if not products:
                QMessageBox.warning(self, "No Products", "No products found in database.")
                return
            
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Product")
            dialog.setMinimumWidth(400)
            
            layout = QVBoxLayout()
            
            # Search box
            search_layout = QHBoxLayout()
            search_label = QLabel("Search:")
            search_layout.addWidget(search_label)
            
            search_input = QLineEdit()
            search_input.setPlaceholderText("Type product Descrip...")
            search_layout.addWidget(search_input)
            
            layout.addLayout(search_layout)
            
            # Product list
            product_list = QListWidget()
            for product_id, Descrip, brand, price in products:
                product_list.addItem(f"{Descrip} - {brand} (₹{price:.2f}) [{product_id}]")
            
            layout.addWidget(product_list)
            
            # Buttons
            button_layout = QHBoxLayout()
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_button)
            
            select_button = QPushButton("Select")
            select_button.clicked.connect(dialog.accept)
            button_layout.addWidget(select_button)
            
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            # Connect search functionality
            def filter_list():
                search_text = search_input.text().lower()
                for i in range(product_list.count()):
                    item = product_list.item(i)
                    item.setHidden(search_text not in item.text().lower())
            
            search_input.textChanged.connect(filter_list)
            
            # Show dialog and process result
            if dialog.exec() == QDialog.DialogCode.Accepted and product_list.currentItem():
                selected_text = product_list.currentItem().text()
                product_id = selected_text.split("[")[-1].split("]")[0]
                
                # Get product details
                cursor.execute("""
                    SELECT Descrip, brand, price, category, type 
                    FROM products 
                    WHERE product_id = ?
                """, (product_id,))
                
                result = cursor.fetchone()
                if result:
                    Descrip, brand, price, category, product_type = result
                    
                    self.current_product_id = product_id
                    self.product_Descrip.setText(Descrip)
                    self.product_price.setText(f"₹ {price:.2f}")
                    self.product_brand.setText(f"{brand} - {category} / {product_type}")
                    self.add_button.setEnabled(True)
                    
                    # Set placeholder image based on category
                    color_map = {
                        'Groceries': '#78e08f',
                        'Beauty': '#a29bfe',
                        'Electronics': '#74b9ff',
                        'Clothing': '#ff7675',
                        'Furniture': '#fdcb6e'
                    }
                    color = color_map.get(category, '#dfe6e9')
                    
                    img = QPixmap(150, 150)
                    img.fill(QColor(color))
                    painter = QPainter(img)
                    painter.setPen(Qt.GlobalColor.white)
                    painter.setFont(QFont('Arial', 12, QFont.Weight.Bold))
                    painter.drawText(QRect(0, 0, 150, 150), Qt.AlignmentFlag.AlignCenter, product_type[0].upper())
                    painter.end()
                    
                    self.product_image.setPixmap(img)
            
        except Exception as e:
            print(f"Error in manual entry: {e}")
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def add_to_cart(self):
        """Add the scanned product to the cart"""
        if self.current_product_id:
            self.scanComplete.emit(self.current_product_id)
            self.add_button.setEnabled(False)
            self.add_button.setText("Added!")
            
            # Reset after a delay
            QTimer.singleShot(2000, lambda: self.add_button.setText("Add to Cart"))

class AllergenDetector:
    """Class to detect allergens in products"""
    def __init__(self):
        pass
    
    def get_user_allergens(self, user_id):
        """Get allergens for a user"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT a.allergen_id, a.Descrip
                FROM allergens a
                JOIN user_allergens ua ON a.allergen_id = ua.allergen_id
                WHERE ua.user_id = ?
            """, (user_id,))
            
            return {row[0]: row[1] for row in cursor.fetchall()}
            
        except Exception as e:
            print(f"Error getting user allergens: {e}")
            return {}
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def check_product_allergens(self, product_id, user_allergens):
        """Check if a product contains allergens the user is allergic to"""
        if not user_allergens:
            return []
            
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get product allergens
            cursor.execute("""
                SELECT a.allergen_id, a.Descrip
                FROM allergens a
                JOIN product_allergens pa ON a.allergen_id = pa.allergen_id
                WHERE pa.product_id = ?
            """, (product_id,))
            
            product_allergens = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Check for common allergens
            common_allergens = []
            for allergen_id, allergen_Descrip in product_allergens.items():
                if allergen_id in user_allergens:
                    common_allergens.append(allergen_Descrip)
            
            return common_allergens
            
        except Exception as e:
            print(f"Error checking product allergens: {e}")
            return []
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def infer_allergens_from_Descrip(self, product_Descrip):
        """Infer potential allergens from product Descrip (fallback method)"""
        allergens = []
        
        # Common allergen keywords
        allergen_keywords = {
            'milk': 'lactose',
            'dairy': 'lactose',
            'cheese': 'lactose',
            'yogurt': 'lactose',
            'butter': 'lactose',
            'cream': 'lactose',
            'wheat': 'gluten',
            'rye': 'gluten',
            'barley': 'gluten',
            'oats': 'gluten',
            'nut': 'nuts',
            'almond': 'nuts',
            'cashew': 'nuts',
            'peanut': 'nuts',
            'fish': 'fish',
            'shellfish': 'shellfish',
            'shrimp': 'shellfish',
            'crab': 'shellfish',
            'lobster': 'shellfish',
            'egg': 'eggs',
            'soy': 'soy',
            'sesame': 'sesame'
        }
        
        product__name__lower = product_Descrip.lower()
        for keyword, allergen in allergen_keywords.items():
            if keyword in product__name__lower:
                allergens.append(allergen)
        
        return list(set(allergens))  # Remove duplicates

class SmartShoppingCartApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Shopping Cart")
        self.setGeometry(100, 100, 1280, 720)
        
        # Set application style
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QPushButton {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {SECONDARY_COLOR};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                padding: 8px;
                border: 1px solid #d1d1d1;
                border-radius: 4px;
                background-color: white;
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QTableWidget {{
                gridline-color: #e0e0e0;
                selection-background-color: {SECONDARY_COLOR};
                selection-color: white;
            }}
            QTabWidget::pane {{
                border: 1px solid #d1d1d1;
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: #e0e0e0;
                padding: 10px 15px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {PRIMARY_COLOR};
                color: white;
            }}
        """)
        
        # Initialize components
        self.current_user_id = None
        self.user_budget = 0
        self.cart_items = []
        self.recommendation_engine = RecommendationEngine()
        self.allergen_detector = AllergenDetector()
        
        # Create the stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create screens
        self.create_login_screen()
        self.create__main__screen()
        
        # Start with login screen
        self.stacked_widget.setCurrentIndex(0)
    
    def create_login_screen(self):
        """Create the login screen"""
        login_widget = QWidget()
        layout = QVBoxLayout()
        
        # Create a centered container for login
        container = QWidget()
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)
        
        # App logo/title
        title_label = QLabel("Smart Shopping Cart")
        title_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {PRIMARY_COLOR};
            margin: 20px 0;
            text-align: center;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(title_label)
        
        # Login form
        login_frame = QFrame()
        login_frame.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            padding: 20px;
        """)
        login_layout = QVBoxLayout()
        
        # UserDescrip
        userDescrip_label = QLabel("UserDescrip")
        userDescrip_label.setStyleSheet("font-weight: bold;")
        login_layout.addWidget(userDescrip_label)
        
        self.userDescrip_input = QLineEdit()
        self.userDescrip_input.setPlaceholderText("Enter your userDescrip")
        login_layout.addWidget(self.userDescrip_input)
        
        # Password
        password_label = QLabel("Password")
        password_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        login_layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        login_layout.addWidget(self.password_input)
        
        # Login button
        login_button = QPushButton("Login")
        login_button.setStyleSheet(f"""
            background-color: {PRIMARY_COLOR};
            color: white;
            padding: 10px;
            font-weight: bold;
            margin-top: 20px;
            height: 40px;
        """)
        login_button.clicked.connect(self.handle_login)
        login_layout.addWidget(login_button)
        
        # Register link
        register_layout = QHBoxLayout()
        register_label = QLabel("Don't have an account?")
        register_layout.addWidget(register_label)
        
        register_button = QPushButton("Register")
        register_button.setStyleSheet("""
            background-color: transparent;
            color: #3498db;
            border: none;
            padding: 0;
            font-weight: bold;
            text-decoration: underline;
        """)
        register_button.clicked.connect(self.show_register_dialog)
        register_layout.addWidget(register_button)
        
        login_layout.addLayout(register_layout)
        
        login_frame.setLayout(login_layout)
        container_layout.addWidget(login_frame)
        
        # Quick login for demo
        demo_button = QPushButton("Quick Login (Demo)")
        demo_button.setStyleSheet("""
            background-color: #95a5a6;
            margin-top: 20px;
        """)
        demo_button.clicked.connect(self.demo_login)
        container_layout.addWidget(demo_button)
        
        # Add container to main layout with centering
        layout.addStretch()
        layout.addWidget(container, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        
        login_widget.setLayout(layout)
        self.stacked_widget.addWidget(login_widget)
    
    def create__main__screen(self):
        """Create the main shopping screen"""
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        
        # Logo
        logo_label = QLabel("Smart Cart")
        logo_label.setStyleSheet(f"""
            font-size: 22px;
            font-weight: bold;
            color: {PRIMARY_COLOR};
        """)
        header_layout.addWidget(logo_label)
        
        header_layout.addStretch()
        
        # Budget display
        self.budget_frame = QFrame()
        self.budget_frame.setStyleSheet(f"""
            background-color: white;
            border-radius: 4px;
            padding: 5px 10px;
        """)
        budget_layout = QHBoxLayout()
        budget_layout.setContentsMargins(10, 5, 10, 5)
        
        budget_label = QLabel("Budget:")
        budget_label.setStyleSheet("font-weight: bold;")
        budget_layout.addWidget(budget_label)
        
        self.budget_value = QLabel("₹0.00")
        budget_layout.addWidget(self.budget_value)
        
        budget_edit_button = QPushButton("Edit")
        budget_edit_button.setStyleSheet("""
            padding: 3px 8px;
            font-size: 12px;
        """)
        budget_edit_button.clicked.connect(self.edit_budget)
        budget_layout.addWidget(budget_edit_button)
        
        self.budget_frame.setLayout(budget_layout)
        header_layout.addWidget(self.budget_frame)
        
        # User info
        self.user_frame = QFrame()
        self.user_frame.setStyleSheet("""
            background-color: white;
            border-radius: 4px;
            padding: 5px 10px;
        """)
        user_layout = QHBoxLayout()
        user_layout.setContentsMargins(10, 5, 10, 5)
        
        self.user_label = QLabel("Guest")
        self.user_label.setStyleSheet("font-weight: bold;")
        user_layout.addWidget(self.user_label)
        
        logout_button = QPushButton("Logout")
        logout_button.setStyleSheet("""
            padding: 3px 8px;
            font-size: 12px;
            background-color: #e74c3c;
        """)
        logout_button.clicked.connect(self.logout)
        user_layout.addWidget(logout_button)
        
        self.user_frame.setLayout(user_layout)
        header_layout.addWidget(self.user_frame)
        
        main_layout.addLayout(header_layout)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left panel (scanner and recommendations)
        left_panel = QWidget()
        left_panel.setMinimumWidth(400)
        left_panel.setMaximumWidth(500)
        left_layout = QVBoxLayout()
        
        # Scanner section
        scanner_label = QLabel("Product Scanner")
        scanner_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        """)
        left_layout.addWidget(scanner_label)
        
        self.scanner = ScannerSimulator()
        self.scanner.scanComplete.connect(self.add_product_to_cart)
        left_layout.addWidget(self.scanner)
        
        # Recommendations section
        recommendations_label = QLabel("Recommended Products")
        recommendations_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            margin: 20px 0 10px 0;
        """)
        left_layout.addWidget(recommendations_label)
        
        self.recommendations_container = QWidget()
        self.recommendations_layout = QVBoxLayout()
        self.recommendations_container.setLayout(self.recommendations_layout)
        
        # Placeholder for recommendations
        self.update_recommendations([])
        
        left_layout.addWidget(self.recommendations_container)
        left_panel.setLayout(left_layout)
        
        # Right panel (cart)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Cart header
        cart_header = QHBoxLayout()
        
        cart_title = QLabel("Your Shopping Cart")
        cart_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)
        cart_header.addWidget(cart_title)
        
        cart_header.addStretch()
        
        clear_cart_button = QPushButton("Clear Cart")
        clear_cart_button.setStyleSheet(f"""
            background-color: {DANGER_COLOR};
            padding: 5px 10px;
        """)
        clear_cart_button.clicked.connect(self.clear_cart)
        cart_header.addWidget(clear_cart_button)
        
        right_layout.addLayout(cart_header)
        
        # Cart items table
        self.cart_table = QTableWidget()
        self.cart_table.setColumnCount(5)
        self.cart_table.setHorizontalHeaderLabels(["Product", "Price", "Qty", "Total", ""])
        self.cart_table.horizontalHeader().setStretchLastSection(False)
        self.cart_table.setColumnWidth(0, 250)  # Product Descrip
        self.cart_table.setColumnWidth(1, 100)  # Price
        self.cart_table.setColumnWidth(2, 50)   # Quantity
        self.cart_table.setColumnWidth(3, 100)  # Total
        self.cart_table.setColumnWidth(4, 80)   # Remove button
        right_layout.addWidget(self.cart_table)
        
        # Total section
        totals_frame = QFrame()
        totals_frame.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            padding: 15px;
        """)
        totals_layout = QVBoxLayout()
        
        # Subtotal
        subtotal_layout = QHBoxLayout()
        subtotal_label = QLabel("Subtotal:")
        subtotal_label.setStyleSheet("font-size: 16px;")
        subtotal_layout.addWidget(subtotal_label)
        subtotal_layout.addStretch()
        self.subtotal_value = QLabel("₹0.00")
        self.subtotal_value.setStyleSheet("font-size: 16px; font-weight: bold;")
        subtotal_layout.addWidget(self.subtotal_value)
        totals_layout.addLayout(subtotal_layout)
        
        # Taxes
        taxes_layout = QHBoxLayout()
        taxes_label = QLabel("Taxes (18%):")
        taxes_layout.addWidget(taxes_label)
        taxes_layout.addStretch()
        self.taxes_value = QLabel("₹0.00")
        taxes_layout.addWidget(self.taxes_value)
        totals_layout.addLayout(taxes_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #ddd;")
        totals_layout.addWidget(separator)
        
        # Total
        total_layout = QHBoxLayout()
        total_label = QLabel("Total:")
        total_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        total_layout.addWidget(total_label)
        total_layout.addStretch()
        self.total_value = QLabel("₹0.00")
        self.total_value.setStyleSheet("font-size: 18px; font-weight: bold; color: #e74c3c;")
        total_layout.addWidget(self.total_value)
        totals_layout.addLayout(total_layout)
        
        # Budget progress
        budget_progress_label = QLabel("Budget Usage:")
        totals_layout.addWidget(budget_progress_label)
        
        self.budget_progress = QProgressBar()
        self.budget_progress.setTextVisible(True)
        self.budget_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4a69bd;
                border-radius: 5px;
            }
        """)
        totals_layout.addWidget(self.budget_progress)
        
        # Checkout button
        checkout_button = QPushButton("Proceed to Checkout")
        checkout_button.setStyleSheet(f"""
            background-color: {SUCCESS_COLOR};
            color: white;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
            margin-top: 10px;
        """)
        checkout_button.clicked.connect(self.checkout)
        totals_layout.addWidget(checkout_button)
        
        totals_frame.setLayout(totals_layout)
        right_layout.addWidget(totals_frame)
        
        right_panel.setLayout(right_layout)
        
        # Add panels to content layout
        content_layout.addWidget(left_panel)
        content_layout.addWidget(right_panel, 1)  # Give right panel more stretch
        
        main_layout.addLayout(content_layout)
        
        main_widget.setLayout(main_layout)
        self.stacked_widget.addWidget(main_widget)
    
    def handle_login(self):
        """Handle user login"""
        userDescrip = self.userDescrip_input.text()
        password = self.password_input.text()
        
        if not userDescrip or not password:
            QMessageBox.warning(self, "Login Error", "Please enter both userDescrip and password.")
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Hash password (in a real app, use proper password hashing)
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            # Check user credentials
            cursor.execute("""
                SELECT user_id, first_Descrip, last_Descrip, budget 
                FROM users 
                WHERE userDescrip = ? AND password = ?
            """, (userDescrip, hashed_password))
            
            result = cursor.fetchone()
            
            if result:
                self.current_user_id, first_Descrip, last_Descrip, budget = result
                self.user_budget = float(budget) if budget else 0
                
                # Update UI
                self.user_label.setText(f"{first_Descrip} {last_Descrip}")
                self.budget_value.setText(f"₹{self.user_budget:.2f}")
                self.update_budget_progress()
                
                # Switch to main screen
                self.stacked_widget.setCurrentIndex(1)
                
                # Load cart if exists
                self.load_cart()
                
                # Show welcome message
                QMessageBox.information(self, "Welcome", f"Welcome back, {first_Descrip}!")
                
            else:
                QMessageBox.warning(self, "Login Failed", "Invalid userDescrip or password.")
                
        except Exception as e:
            print(f"Login error: {e}")
            QMessageBox.critical(self, "Error", "An error occurred during login.")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def show_register_dialog(self):
        """Show registration dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Register New Account")
        dialog.setFixedWidth(400)
        
        layout = QVBoxLayout()
        
        # Form fields
        form_layout = QVBoxLayout()
        
        # UserDescrip
        userDescrip_label = QLabel("UserDescrip:")
        form_layout.addWidget(userDescrip_label)
        userDescrip_input = QLineEdit()
        form_layout.addWidget(userDescrip_input)
        
        # Password
        password_label = QLabel("Password:")
        form_layout.addWidget(password_label)
        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.EchoMode.Password)
        form_layout.addWidget(password_input)
        
        # Confirm Password
        confirm_label = QLabel("Confirm Password:")
        form_layout.addWidget(confirm_label)
        confirm_input = QLineEdit()
        confirm_input.setEchoMode(QLineEdit.EchoMode.Password)
        form_layout.addWidget(confirm_input)
        
        # Email
        email_label = QLabel("Email:")
        form_layout.addWidget(email_label)
        email_input = QLineEdit()
        form_layout.addWidget(email_input)
        
        # First Name
        fDescrip_label = QLabel("First Name:")
        form_layout.addWidget(fDescrip_label)
        fDescrip_input = QLineEdit()
        form_layout.addWidget(fDescrip_input)
        
        # Last Name
        lDescrip_label = QLabel("Last Name:")
        form_layout.addWidget(lDescrip_label)
        lDescrip_input = QLineEdit()
        form_layout.addWidget(lDescrip_input)
        
        # Budget
        budget_label = QLabel("Shopping Budget (₹):")
        form_layout.addWidget(budget_label)
        budget_input = QDoubleSpinBox()
        budget_input.setRange(0, 100000)
        budget_input.setSingleStep(100)
        budget_input.setValue(2000)
        form_layout.addWidget(budget_input)
        
        # Allergens
        allergens_label = QLabel("Select Your Allergens:")
        form_layout.addWidget(allergens_label)
        
        allergens_container = QWidget()
        allergens_layout = QVBoxLayout()
        allergens_container.setLayout(allergens_layout)
        
        # Common allergens
        common_allergens = ["Milk/Lactose", "Nuts", "Gluten", "Eggs", "Soy", "Fish", "Shellfish"]
        allergen_checkboxes = {}
        
        for allergen in common_allergens:
            checkbox = QCheckBox(allergen)
            allergens_layout.addWidget(checkbox)
            allergen_checkboxes[allergen] = checkbox
        
        form_layout.addWidget(allergens_container)
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        register_button = QPushButton("Register")
        register_button.setStyleSheet(f"background-color: {PRIMARY_COLOR};")
        register_button.clicked.connect(lambda: self.register_user(
            userDescrip_input.text(),
            password_input.text(),
            confirm_input.text(),
            email_input.text(),
            fDescrip_input.text(),
            lDescrip_input.text(),
            budget_input.value(),
            {k: v.isChecked() for k, v in allergen_checkboxes.items()},
            dialog
        ))
        button_layout.addWidget(register_button)
        
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def register_user(self, userDescrip, password, confirm_password, email, first_Descrip, last_Descrip, budget, allergens, dialog):
        """Register a new user"""
        # Validate inputs
        if not userDescrip or not password or not email or not first_Descrip:
            QMessageBox.warning(self, "Registration Error", "Please fill all required fields.")
            return
        
        if password != confirm_password:
            QMessageBox.warning(self, "Registration Error", "Passwords do not match.")
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if userDescrip already exists
            cursor.execute("SELECT user_id FROM users WHERE userDescrip = ?", (userDescrip,))
            if cursor.fetchone():
                QMessageBox.warning(self, "Registration Error", "UserDescrip already exists.")
                return
            
            # Check if email already exists
            cursor.execute("SELECT user_id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                QMessageBox.warning(self, "Registration Error", "Email already exists.")
                return
            
            # Hash password (in a real app, use proper password hashing)
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            # Insert user
            cursor.execute("""
                INSERT INTO users (userDescrip, password, email, first_Descrip, last_Descrip, budget)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (userDescrip, hashed_password, email, first_Descrip, last_Descrip, budget))
            
            # Get user ID
            user_id = cursor.lastrowid
            
            # Insert user allergens
            for allergen_Descrip, is_checked in allergens.items():
                if is_checked:
                    # Check if allergen exists
                    cursor.execute("SELECT allergen_id FROM allergens WHERE Descrip = ?", (allergen_Descrip.lower(),))
                    result = cursor.fetchone()
                    
                    if result:
                        allergen_id = result[0]
                    else:
                        # Create new allergen
                        cursor.execute("INSERT INTO allergens (Descrip) VALUES (?)", (allergen_Descrip.lower(),))
                        allergen_id = cursor.lastrowid
                    
                    # Link allergen to user
                    cursor.execute("""
                        INSERT INTO user_allergens (user_id, allergen_id)
                        VALUES (?, ?)
                    """, (user_id, allergen_id))
            
            conn.commit()
            
            # Show success message
            QMessageBox.information(self, "Registration Successful", 
                                   f"Welcome, {first_Descrip}! Your account has been created.")
            
            # Pre-fill login fields
            self.userDescrip_input.setText(userDescrip)
            self.password_input.setText(password)
            
            # Close dialog
            dialog.accept()
            
        except Exception as e:
            print(f"Registration error: {e}")
            QMessageBox.critical(self, "Error", "An error occurred during registration.")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def demo_login(self):
        """Quick login for demo purposes"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if demo user exists
            cursor.execute("SELECT user_id FROM users WHERE userDescrip = 'demo'")
            result = cursor.fetchone()
            
            if result:
                # Demo user exists, use it
                self.userDescrip_input.setText("demo")
                self.password_input.setText("demo123")
                self.handle_login()
            else:
                # Create demo user
                hashed_password = hashlib.sha256("demo123".encode()).hexdigest()
                
                cursor.execute("""
                    INSERT INTO users (userDescrip, password, email, first_Descrip, last_Descrip, budget)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("demo", hashed_password, "demo@example.com", "Demo", "User", 2000))
                
                # Add some allergies
                user_id = cursor.lastrowid
                
                # Make sure allergens exist
                common_allergens = ["nuts", "lactose"]
                for allergen_Descrip in common_allergens:
                    cursor.execute("SELECT allergen_id FROM allergens WHERE Descrip = ?", (allergen_Descrip,))
                    result = cursor.fetchone()
                    
                    if result:
                        allergen_id = result[0]
                    else:
                        cursor.execute("INSERT INTO allergens (Descrip) VALUES (?)", (allergen_Descrip,))
                        allergen_id = cursor.lastrowid
                    
                    # Link allergen to user
                    cursor.execute("""
                        INSERT INTO user_allergens (user_id, allergen_id)
                        VALUES (?, ?)
                    """, (user_id, allergen_id))
                
                conn.commit()
                
                self.userDescrip_input.setText("demo")
                self.password_input.setText("demo123")
                self.handle_login()
                
        except Exception as e:
            print(f"Demo login error: {e}")
            QMessageBox.critical(self, "Error", "An error occurred during demo login.")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def logout(self):
        """Log out the current user"""
        self.current_user_id = None
        self.user_budget = 0
        self.cart_items = []
        
        # Clear UI
        self.cart_table.setRowCount(0)
        self.update_cart_totals()
        self.update_recommendations([])
        
        # Switch to login screen
        self.stacked_widget.setCurrentIndex(0)
        self.userDescrip_input.clear()
        self.password_input.clear()
    
    def edit_budget(self):
        """Edit user budget"""
        if not self.current_user_id:
            return
        
        # Show dialog to edit budget
        new_budget, ok = QInputDialog.getDouble(
            self, "Edit Budget", "Enter new budget (₹):", 
            self.user_budget, 0, 100000, 2
        )
        
        if ok:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Update budget
                cursor.execute("""
                    UPDATE users SET budget = ? WHERE user_id = ?
                """, (new_budget, self.current_user_id))
                
                conn.commit()
                
                # Update UI
                self.user_budget = new_budget
                self.budget_value.setText(f"₹{self.user_budget:.2f}")
                self.update_budget_progress()
                
                # Update recommendations based on new budget
                self.update_recommendations_display()
                
            except Exception as e:
                print(f"Error updating budget: {e}")
                QMessageBox.critical(self, "Error", "An error occurred while updating your budget.")
                
            finally:
                if conn:
                    cursor.close()
                    conn.close()
    
    def load_cart(self):
        """Load shopping cart for current user"""
        if not self.current_user_id:
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get active cart
            cursor.execute("""
                SELECT cart_id FROM cart 
                WHERE user_id = ? AND status = 'active'
                ORDER BY creation_date DESC
                LIMIT 1
            """, (self.current_user_id,))
            
            result = cursor.fetchone()
            
            if result:
                cart_id = result[0]
                
                # Get cart items
                cursor.execute("""
                    SELECT ci.product_id, p.Descrip, p.price, ci.quantity
                    FROM cart_items ci
                    JOIN products p ON ci.product_id = p.product_id
                    WHERE ci.cart_id = ?
                """, (cart_id,))
                
                cart_items = cursor.fetchall()
                
                # Clear cart
                self.cart_items = []
                self.cart_table.setRowCount(0)
                
                # Add items to cart
                for product_id, Descrip, price, quantity in cart_items:
                    self.add_product_to_cart(product_id, quantity)
                
            else:
                # Create new cart
                cursor.execute("""
                    INSERT INTO cart (user_id, status)
                    VALUES (?, 'active')
                """, (self.current_user_id,))
                
                conn.commit()
            
        except Exception as e:
            print(f"Error loading cart: {e}")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def add_product_to_cart(self, product_id, quantity=1):
        """Add a product to the cart"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get product details
            cursor.execute("""
                SELECT Descrip, price, category, type, brand
                FROM products
                WHERE product_id = ?
            """, (product_id,))
            
            result = cursor.fetchone()
            
            if not result:
                print(f"Product not found: {product_id}")
                return
            
            Descrip, price, category, product_type, brand = result
            
            # Check if product is already in cart
            for i, item in enumerate(self.cart_items):
                if item['product_id'] == product_id:
                    # Update quantity
                    item['quantity'] += quantity
                    
                    # Update table
                    qty_spinbox = self.cart_table.cellWidget(i, 2)
                    qty_spinbox.setValue(item['quantity'])
                    
                    # Update total
                    total = item['price'] * item['quantity']
                    self.cart_table.setItem(i, 3, QTableWidgetItem(f"₹{total:.2f}"))
                    
                    break
            else:
                # Product not in cart, add it
                item = {
                    'product_id': product_id,
                    'price': float(price),
                }
                
                self.cart_items.append(item)
                
                # Add to table
                row = self.cart_table.rowCount()
                self.cart_table.insertRow(row)
                
                # Product Descrip
                self.cart_table.setItem(row, 0, QTableWidgetItem(Descrip))
                
                # Price
                self.cart_table.setItem(row, 1, QTableWidgetItem(f"₹{float(price):.2f}"))
                
                # Quantity
                qty_spinbox = QSpinBox()
                qty_spinbox.setRange(1, 99)
                qty_spinbox.setValue(quantity)
                qty_spinbox.valueChanged.connect(lambda val, r=row: self.update_item_quantity(r, val))
                self.cart_table.setCellWidget(row, 2, qty_spinbox)
                
                # Total
                total = float(price) * quantity
                self.cart_table.setItem(row, 3, QTableWidgetItem(f"₹{total:.2f}"))
                
                # Remove button
                remove_button = QPushButton("🗑️")
                remove_button.setStyleSheet("""
                    background-color: transparent;
                    color: #e74c3c;
                    border: none;
                    font-size: 16px;
                    font-weight: bold;
                """)
                remove_button.clicked.connect(lambda _, r=row: self.remove_item(r))
                
                button_container = QWidget()
                button_layout = QHBoxLayout()
                button_layout.setContentsMargins(0, 0, 0, 0)
                button_layout.addWidget(remove_button)
                button_container.setLayout(button_layout)
                
                self.cart_table.setCellWidget(row, 4, button_container)
            
            # Check for allergens
            if self.current_user_id:
                user_allergens = self.allergen_detector.get_user_allergens(self.current_user_id)
                allergens = self.allergen_detector.check_product_allergens(product_id, user_allergens)
                
                if not allergens:
                    # Try inferring from Descrip
                    allergens = self.allergen_detector.infer_allergens_from_Descrip(Descrip)
                    allergens = [a for a in allergens if a in user_allergens.values()]
                
                if allergens:
                    QMessageBox.warning(
                        self,
                        "Allergen Alert",
                        f"Warning: This product may contain allergens you're sensitive to: {', '.join(allergens)}."
                    )
            
            # Save to database
            if self.current_user_id:
                # Get active cart
                cursor.execute("""
                    SELECT cart_id FROM cart 
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY creation_date DESC
                    LIMIT 1
                """, (self.current_user_id,))
                
                result = cursor.fetchone()
                
                if result:
                    cart_id = result[0]
                    
                    # Check if item exists in cart
                    cursor.execute("""
                        SELECT quantity FROM cart_items
                        WHERE cart_id = ? AND product_id = ?
                    """, (cart_id, product_id))
                    
                    result = cursor.fetchone()
                    
                    if result:
                        # Update quantity
                        current_quantity = result[0]
                        new_quantity = current_quantity + quantity
                        
                        cursor.execute("""
                            UPDATE cart_items
                            SET quantity = ?
                            WHERE cart_id = ? AND product_id = ?
                        """, (new_quantity, cart_id, product_id))
                    else:
                        # Add new item
                        cursor.execute("""
                            INSERT INTO cart_items (cart_id, product_id, quantity)
                            VALUES (?, ?, ?)
                        """, (cart_id, product_id, quantity))
                    
                    conn.commit()
            
            # Update totals
            self.update_cart_totals()
            
            # Update recommendations
            self.update_recommendations_display()
            
        except Exception as e:
            print(f"Error adding product to cart: {e}")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def update_item_quantity(self, row, quantity):
        """Update quantity of an item in the cart"""
        if row < 0 or row >= len(self.cart_items):
            return
        
        # Update cart item
        self.cart_items[row]['quantity'] = quantity
        
        # Update total
        price = self.cart_items[row]['price']
        total = price * quantity
        self.cart_table.setItem(row, 3, QTableWidgetItem(f"₹{total:.2f}"))
        
        # Update database
        if self.current_user_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Get active cart
                cursor.execute("""
                    SELECT cart_id FROM cart 
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY creation_date DESC
                    LIMIT 1
                """, (self.current_user_id,))
                
                result = cursor.fetchone()
                
                if result:
                    cart_id = result[0]
                    product_id = self.cart_items[row]['product_id']
                    
                    # Update quantity
                    cursor.execute("""
                        UPDATE cart_items
                        SET quantity = ?
                        WHERE cart_id = ? AND product_id = ?
                    """, (quantity, cart_id, product_id))
                    
                    conn.commit()
                
            except Exception as e:
                print(f"Error updating item quantity: {e}")
                
            finally:
                if conn:
                    cursor.close()
                    conn.close()
        
        # Update totals
        self.update_cart_totals()
        
        # Update recommendations
        self.update_recommendations_display()
    
    def remove_item(self, row):
        """Remove item from cart"""
        if row < 0 or row >= len(self.cart_items):
            return
        
        product_id = self.cart_items[row]['product_id']
        
        # Remove from database
        if self.current_user_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Get active cart
                cursor.execute("""
                    SELECT cart_id FROM cart 
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY creation_date DESC
                    LIMIT 1
                """, (self.current_user_id,))
                
                result = cursor.fetchone()
                
                if result:
                    cart_id = result[0]
                    
                    # Remove item
                    cursor.execute("""
                        DELETE FROM cart_items
                        WHERE cart_id = ? AND product_id = ?
                    """, (cart_id, product_id))
                    
                    conn.commit()
                
            except Exception as e:
                print(f"Error removing item: {e}")
                
            finally:
                if conn:
                    cursor.close()
                    conn.close()
        
        # Remove from cart
        self.cart_items.pop(row)
        self.cart_table.removeRow(row)
        
        # Update totals
        self.update_cart_totals()
        
        # Update recommendations
        self.update_recommendations_display()
    
    def clear_cart(self):
        """Clear all items from cart"""
        if not self.cart_items:
            return
        
        # Confirm
        reply = QMessageBox.question(
            self, "Clear Cart", 
            "Are you sure you want to clear your cart?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Clear from database
        if self.current_user_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Get active cart
                cursor.execute("""
                    SELECT cart_id FROM cart 
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY creation_date DESC
                    LIMIT 1
                """, (self.current_user_id,))
                
                result = cursor.fetchone()
                
                if result:
                    cart_id = result[0]
                    
                    # Remove all items
                    cursor.execute("""
                        DELETE FROM cart_items
                        WHERE cart_id = ?
                    """, (cart_id,))
                    
                    conn.commit()
                
            except Exception as e:
                print(f"Error clearing cart: {e}")
                
            finally:
                if conn:
                    cursor.close()
                    conn.close()
        
        # Clear cart
        self.cart_items = []
        self.cart_table.setRowCount(0)
        
        # Update totals
        self.update_cart_totals()
        
        # Update recommendations
        self.update_recommendations_display()
    
    def update_cart_totals(self):
        """Update cart total display"""
        subtotal = sum(item['price'] * item['quantity'] for item in self.cart_items)
        taxes = subtotal * 0.18  # 18% tax rate
        total = subtotal + taxes
        
        self.subtotal_value.setText(f"₹{subtotal:.2f}")
        self.taxes_value.setText(f"₹{taxes:.2f}")
        self.total_value.setText(f"₹{total:.2f}")
        
        # Update budget progress
        self.update_budget_progress(total)
    
    def update_budget_progress(self, total=0):
        """Update budget progress bar"""
        if self.user_budget <= 0:
            self.budget_progress.setValue(0)
            self.budget_progress.setFormat("No budget set")
            return
        
        percentage = (total / self.user_budget) * 100
        self.budget_progress.setValue(int(percentage))
        
        if percentage <= 75:
            self.budget_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #78e08f;
                    border-radius: 5px;
                }
            """)
        elif percentage <= 90:
            self.budget_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #f6b93b;
                    border-radius: 5px;
                }
            """)
        else:
            self.budget_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #e55039;
                    border-radius: 5px;
                }
            """)
        
        self.budget_progress.setFormat(f"{percentage:.1f}% of budget (₹{total:.2f} / ₹{self.user_budget:.2f})")
        
        # Show warning if over budget
        if percentage > 100 and total > 0:
            QMessageBox.warning(
                self,
                "Budget Alert",
                f"Your current total (₹{total:.2f}) exceeds your budget (₹{self.user_budget:.2f}).\n\n"
                "Would you like to see suggestions for cheaper alternatives?"
            )
            
            # Show budget alternatives
            self.show_budget_alternatives()
    
    def show_budget_alternatives(self):
        """Show budget-friendly alternatives"""
        if not self.cart_items or self.user_budget <= 0:
            return
        
        try:
            # Calculate current total
            subtotal = sum(item['price'] * item['quantity'] for item in self.cart_items)
            total = subtotal * 1.18  # Including tax
            
            # If under budget, do nothing
            if total <= self.user_budget:
                return
            
            # Sort items by price (highest first)
            sorted_items = sorted(self.cart_items, key=lambda x: x['price'] * x['quantity'], reverse=True)
            
            # Find most expensive items
            expensive_items = sorted_items[:3]
            
            # Calculate budget overage
            budget_overage = total - self.user_budget
            
            # Get alternatives
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            alternatives = []
            
            for item in expensive_items:
                product_id = item['product_id']
                category = item['category']
                product_type = item['type']
                current_price = item['price']
                
                # Find cheaper alternatives
                cursor.execute("""
                    SELECT p.product_id, p.Descrip, p.brand, p.price
                    FROM products p
                    WHERE p.category = ? 
                      AND p.type = ? 
                      AND p.price < ?
                      AND p.product_id != ?
                    ORDER BY p.price DESC
                    LIMIT 3
                """, (category, product_type, current_price, product_id))
                
                results = cursor.fetchall()
                
                if results:
                    alternatives.append({
                        'original': item,
                        'alternatives': [
                            {
                                'product_id': r[0],
                                'price': float(r[3])
                            }
                            for r in results
                        ]
                    })
            
            # Show alternatives dialog
            if alternatives:
                dialog = QDialog(self)
                dialog.setWindowTitle("Budget-Friendly Alternatives")
                dialog.setMinimumWidth(600)
                
                layout = QVBoxLayout()
                
                # Heading
                heading = QLabel("Suggested alternatives to stay within your budget:")
                heading.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
                layout.addWidget(heading)
                
                # Budget info
                budget_info = QLabel(f"Current total: ₹{total:.2f}  |  Your budget: ₹{self.user_budget:.2f}  |  Over by: ₹{budget_overage:.2f}")
                budget_info.setStyleSheet(f"color: {DANGER_COLOR}; margin-bottom: 20px;")
                layout.addWidget(budget_info)
                
                # Alternatives
                for alt in alternatives:
                    original = alt['original']
                    
                    # Create frame
                    item_frame = QFrame()
                    item_frame.setStyleSheet("""
                        background-color: white;
                        border-radius: 8px;
                        padding: 10px;
                        margin-bottom: 15px;
                    """)
                    item_layout = QVBoxLayout()
                    
                    # Original item
                    original_label = QLabel(f"Original: {original['Descrip']} ({original['brand']}) - ₹{original['price']:.2f}")
                    original_label.setStyleSheet("font-weight: bold;")
                    item_layout.addWidget(original_label)
                    
                    # Alternatives
                    for i, alternative in enumerate(alt['alternatives']):
                        savings = (original['price'] - alternative['price']) * original['quantity']
                        
                        alt_frame = QFrame()
                        alt_frame.setStyleSheet("""
                            background-color: #f0f0f0;
                            border-radius: 4px;
                            padding: 8px;
                            margin: 5px 0;
                        """)
                        alt_layout = QHBoxLayout()
                        
                        alt_label = QLabel(f"{alternative['Descrip']} ({alternative['brand']}) - ₹{alternative['price']:.2f}")
                        alt_label.setStyleSheet("flex: 1;")
                        alt_layout.addWidget(alt_label)
                        
                        savings_label = QLabel(f"Save: ₹{savings:.2f}")
                        savings_label.setStyleSheet(f"color: {SUCCESS_COLOR}; font-weight: bold;")
                        alt_layout.addWidget(savings_label)
                        
                        switch_button = QPushButton("Switch")
                        switch_button.setStyleSheet(f"""
                            background-color: {PRIMARY_COLOR};
                            color: white;
                            padding: 5px 10px;
                            border-radius: 4px;
                        """)
                        switch_button.clicked.connect(
                            lambda _, orig=original, alt=alternative: 
                            self.switch_to_alternative(orig, alt, dialog)
                        )
                        alt_layout.addWidget(switch_button)
                        
                        alt_frame.setLayout(alt_layout)
                        item_layout.addWidget(alt_frame)
                    
                    item_frame.setLayout(item_layout)
                    layout.addWidget(item_frame)
                
                # Close button
                close_button = QPushButton("Close")
                close_button.clicked.connect(dialog.accept)
                layout.addWidget(close_button)
                
                dialog.setLayout(layout)
                dialog.exec()
                
        except Exception as e:
            print(f"Error showing budget alternatives: {e}")
            
        finally:
            if 'conn' in locals() and conn:
                cursor.close()
                conn.close()
    
    def switch_to_alternative(self, original, alternative, dialog):
        """Switch an item to a cheaper alternative"""
        try:
            # Find original item in cart
            for i, item in enumerate(self.cart_items):
                if item['product_id'] == original['product_id']:
                    # Remove original item
                    self.remove_item(i)
                    
                    # Add alternative
                    self.add_product_to_cart(alternative['product_id'], original['quantity'])
                    
                    # Show confirmation
                    QMessageBox.information(
                        self,
                        "Alternative Added",
                        f"Switched to {alternative['Descrip']} and saved ₹{(original['price'] - alternative['price']) * original['quantity']:.2f}!"
                    )
                    
                    # Close dialog
                    dialog.accept()
                    break
            
        except Exception as e:
            print(f"Error switching to alternative: {e}")
    
    def update_recommendations_display(self):
        """Update product recommendations display"""
        if not self.cart_items:
            self.update_recommendations([])
            return
        
        # Get product IDs from cart
        product_ids = [item['product_id'] for item in self.cart_items]
        
        # Get recommendations
        recommended_ids = self.recommendation_engine.get_recommendations(
            product_ids,
            self.current_user_id,
            self.user_budget
        )
        
        # Update display
        self.update_recommendations(recommended_ids)
    
    def update_recommendations(self, product_ids):
        """Update recommendations UI with product IDs"""
        # Clear existing recommendations
        for i in reversed(range(self.recommendations_layout.count())):
            widget = self.recommendations_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        if not product_ids:
            # No recommendations
            empty_label = QLabel("No recommendations yet. Add items to your cart to see suggestions.")
            empty_label.setStyleSheet("""
                color: #7f8c8d;
                padding: 20px;
                text-align: center;
            """)
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.recommendations_layout.addWidget(empty_label)
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get product details
            placeholders = ','.join(['?' for _ in product_ids])
            cursor.execute(f"""
                SELECT product_id, Descrip, brand, price, category, type
                FROM products
                WHERE product_id IN ({placeholders})
            """, product_ids)
            
            products = cursor.fetchall()
            
            # Add product recommendations
            for product_id, Descrip, brand, price, category, product_type in products:
                # Create product card
                card = QFrame()
                card.setStyleSheet("""
                    background-color: white;
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 10px;
                """)
                card_layout = QHBoxLayout()
                
                # Product icon based on category
                icon_label = QLabel()
                icon_label.setFixedSize(50, 50)
                
                color_map = {
                    'Groceries': '#78e08f',
                    'Beauty': '#a29bfe',
                    'Electronics': '#74b9ff',
                    'Clothing': '#ff7675',
                    'Furniture': '#fdcb6e'
                }
                color = color_map.get(category, '#dfe6e9')
                
                icon = QPixmap(50, 50)
                icon.fill(QColor(color))
                
                painter = QPainter(icon)
                painter.setPen(Qt.GlobalColor.white)
                painter.setFont(QFont('Arial', 12, QFont.Weight.Bold))
                painter.drawText(QRect(0, 0, 50, 50), Qt.AlignmentFlag.AlignCenter, product_type[0].upper())
                painter.end()
                
                icon_label.setPixmap(icon)
                card_layout.addWidget(icon_label)
                
                # Product info
                info_layout = QVBoxLayout()
                
                Descrip_label = QLabel(Descrip)
                Descrip_label.setStyleSheet("font-weight: bold; font-size: 14px;")
                info_layout.addWidget(Descrip_label)
                
                brand_label = QLabel(brand)
                brand_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
                info_layout.addWidget(brand_label)
                
                price_label = QLabel(f"₹{float(price):.2f}")
                price_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
                info_layout.addWidget(price_label)
                
                card_layout.addLayout(info_layout)
                
                # Add button
                add_button = QPushButton("Add")
                add_button.setStyleSheet(f"""
                    background-color: {SUCCESS_COLOR};
                    color: white;
                    border-radius: 4px;
                    padding: 5px 10px;
                    font-weight: bold;
                """)
                add_button.setFixedWidth(60)
                add_button.clicked.connect(lambda _, pid=product_id: self.add_product_to_cart(pid))
                card_layout.addWidget(add_button)
                
                card.setLayout(card_layout)
                self.recommendations_layout.addWidget(card)
            
        except Exception as e:
            print(f"Error updating recommendations: {e}")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def checkout(self):
        """Process checkout"""
        if not self.cart_items:
            QMessageBox.warning(self, "Empty Cart", "Your cart is empty.")
            return
        
        # Calculate total
        subtotal = sum(item['price'] * item['quantity'] for item in self.cart_items)
        taxes = subtotal * 0.18
        total = subtotal + taxes
        
        # Check budget
        if self.user_budget > 0 and total > self.user_budget:
            reply = QMessageBox.question(
                self, "Budget Exceeded", 
                f"Your total (₹{total:.2f}) exceeds your budget (₹{self.user_budget:.2f}).\n\n"
                "Would you like to see suggestions for cheaper alternatives before checkout?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel, 
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.show_budget_alternatives()
                return
            elif reply == QMessageBox.StandardButton.Cancel:
                return
        
        # Confirm checkout
        items_text = "\n".join([f"• {item['Descrip']} (x{item['quantity']}) - ₹{item['price'] * item['quantity']:.2f}" 
                               for item in self.cart_items])
        
        confirm_msg = f"Please confirm your order:\n\n{items_text}\n\nSubtotal: ₹{subtotal:.2f}\nTaxes: ₹{taxes:.2f}\n\nTotal: ₹{total:.2f}"
        
        reply = QMessageBox.question(
            self, "Confirm Order", 
            confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Process order
        success = False
        
        if self.current_user_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Get active cart
                cursor.execute("""
                    SELECT cart_id FROM cart 
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY creation_date DESC
                    LIMIT 1
                """, (self.current_user_id,))
                
                result = cursor.fetchone()
                
                if result:
                    cart_id = result[0]
                    
                    # Create purchase record
                    cursor.execute("""
                        INSERT INTO purchase_history (user_id, total_amount)
                        VALUES (?, ?)
                    """, (self.current_user_id, total))
                    
                    purchase_id = cursor.lastrowid
                    
                    # Add purchase items
                    for item in self.cart_items:
                        cursor.execute("""
                            INSERT INTO purchase_items (purchase_id, product_id, quantity, price_at_purchase)
                            VALUES (?, ?, ?, ?)
                        """, (purchase_id, item['product_id'], item['quantity'], item['price']))
                    
                    # Mark cart as checked out
                    cursor.execute("""
                        UPDATE cart SET status = 'checkout'
                        WHERE cart_id = ?
                    """, (cart_id,))
                    
                    # Create new cart
                    cursor.execute("""
                        INSERT INTO cart (user_id, status)
                        VALUES (?, 'active')
                    """, (self.current_user_id,))
                    
                    conn.commit()
                    success = True
                
            except Exception as e:
                print(f"Error processing order: {e}")
                QMessageBox.critical(self, "Error", f"An error occurred during checkout: {str(e)}")
                
            finally:
                if conn:
                    cursor.close()
                    conn.close()
        else:
            # Guest checkout
            success = True
        
        if success:
            # Show receipt
            self.show_receipt(total)
            
            # Clear cart
            self.cart_items = []
            self.cart_table.setRowCount(0)
            self.update_cart_totals()
            
            # Update recommendations
            self.update_recommendations([])
            
            QMessageBox.information(self, "Order Complete", 
                                   "Your order has been processed successfully!")
    
    def show_receipt(self, total):
        """Display receipt"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Receipt")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Store info
        store_label = QLabel("Smart Shopping Cart")
        store_label.setStyleSheet("font-size: 18px; font-weight: bold; text-align: center;")
        store_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(store_label)
        
        date_label = QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(date_label)
        
        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator1)
        
        # Items
        items_label = QLabel("Items purchased:")
        items_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(items_label)
        
        for item in self.cart_items:
            item_label = QLabel(f"{item['Descrip']} (x{item['quantity']})")
            layout.addWidget(item_label)
            
            price_label = QLabel(f"  ₹{item['price']:.2f} each - ₹{item['price'] * item['quantity']:.2f}")
            price_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            layout.addWidget(price_label)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator2)
        
        # Totals
        subtotal = sum(item['price'] * item['quantity'] for item in self.cart_items)
        taxes = subtotal * 0.18
        
        subtotal_label = QLabel(f"Subtotal: ₹{subtotal:.2f}")
        subtotal_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(subtotal_label)
        
        taxes_label = QLabel(f"Taxes (18%): ₹{taxes:.2f}")
        taxes_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(taxes_label)
        
        total_label = QLabel(f"Total: ₹{total:.2f}")
        total_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        total_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(total_label)
        
        # Thank you
        thanks_label = QLabel("Thank you for shopping with us!")
        thanks_label.setStyleSheet("margin-top: 20px; text-align: center;")
        thanks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(thanks_label)
        
        # Save button
        save_button = QPushButton("Save Receipt")
        save_button.clicked.connect(lambda: self.save_receipt_pdf(dialog))
        layout.addWidget(save_button)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def save_receipt_pdf(self, dialog):
        """Save receipt as PDF (simplified)"""
        fileDescrip, _ = QFileDialog.getSaveFileName(
            dialog, "Save Receipt", "", "PDF Files (*.pdf);;All Files (*)"
        )
        
        if not fileDescrip:
            return
        
        try:
            # In a full implementation, we'd generate a PDF here
            # For this demo, we'll just show a confirmation
            QMessageBox.information(dialog, "Receipt Saved", 
                                  f"Receipt has been saved to {fileDescrip}")
            
        except Exception as e:
            print(f"Error saving receipt: {e}")
            QMessageBox.critical(dialog, "Error", f"An error occurred while saving the receipt: {str(e)}")

# ================ APPLICATION ENTRY POINT ================

def main():
    """Main application entry point"""
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
    
    # Show splash screen
    splash_pixmap = QPixmap(400, 300)
    splash_pixmap.fill(QColor(PRIMARY_COLOR))
    
    painter = QPainter(splash_pixmap)
    painter.setPen(Qt.GlobalColor.white)
    painter.setFont(QFont('Arial', 24, QFont.Weight.Bold))
    painter.drawText(QRect(0, 100, 400, 50), Qt.AlignmentFlag.AlignCenter, "Smart Shopping Cart")
    painter.setFont(QFont('Arial', 12))
    painter.drawText(QRect(0, 150, 400, 50), Qt.AlignmentFlag.AlignCenter, "Loading...")
    painter.end()
    
    splash = QSplashScreen(splash_pixmap)
    splash.show()
    app.processEvents()
    
    # Initialize database
    create_database_tables()
    
    # Import sample data
    try:
        # Try to get dataset paths from command line arguments
        train_dataset = sys.argv[1] if len(sys.argv) > 1 else None
        test_dataset = sys.argv[2] if len(sys.argv) > 2 else None
        
        if train_dataset and os.path.exists(train_dataset):
            print(f"Importing training dataset from {train_dataset}")
            import_dataset(train_dataset)
        
        if test_dataset and os.path.exists(test_dataset):
            print(f"Importing test dataset from {test_dataset}")
            import_dataset(test_dataset)
            
    except Exception as e:
        print(f"Error importing datasets: {e}")
    
    # Create main window
    time.sleep(1)  # Simulate loading time
    window = SmartShoppingCartApp()
    window.show()
    splash.finish(window)
    
    # Run application
    sys.exit(app.exec())

if ___name___ == "___main___":
    main()