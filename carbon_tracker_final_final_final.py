import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import os
import tkinter as tk
from tkinter import messagebox

# Constants
DATA_FILE = "carbon_data.csv"
EMISSION_FACTORS = {
    "car_km": 0.21,
    "bus_km": 0.10,
    "bike_km": 0.0,
    "electricity_kwh": 0.45,
    "meat_meal": 5.0,
    "vegetarian_meal": 2.0
}

# Ensure the CSV file exists
def initialize_data_file():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=[
            "date", "car_km", "bus_km", "bike_km", 
            "electricity_kwh", "meat_meal", "vegetarian_meal"
        ])
        df.to_csv(DATA_FILE, index=False)

# Calculate carbon footprint for a single entry
def calculate_footprint(entry):
    return round(sum(entry[k] * EMISSION_FACTORS[k] for k in EMISSION_FACTORS), 2)

# Save a new entry to CSV
def save_data(entry):
    df = pd.read_csv(DATA_FILE)
    new_row = pd.DataFrame([entry])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# Train ML model
def train_model(df):
    df = df.copy()
    df["total_emissions"] = df.apply(lambda row: calculate_footprint(row), axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df["date_ordinal"] = df["date"].map(datetime.date.toordinal)

    X = df[["date_ordinal"]]
    y = df["total_emissions"]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return model, df, y_pred, mae, rmse, r2

# Predict next day emission
def predict_next_day(model, last_date):
    next_date = last_date + pd.Timedelta(days=1)
    next_ordinal = next_date.toordinal()
    next_pred = model.predict([[next_ordinal]])[0]
    return next_date, round(next_pred, 2)

# View chart and prediction
def show_summary():
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        messagebox.showinfo("Info", "No data to show!")
        return

    model, df, y_pred, mae, rmse, r2 = train_model(df)
    next_date, next_pred = predict_next_day(model, df["date"].max())

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["total_emissions"], label="Actual", marker="o")
    plt.plot(df["date"], y_pred, label="Predicted", linestyle="--")
    plt.scatter(next_date, next_pred, color='red', label="Next Day Prediction")
    plt.title("Daily Carbon Footprint - Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("CO2 Emissions (kg)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    messagebox.showinfo("Model Evaluation",
        f"📊 MAE: {mae:.2f} kg\n📉 RMSE: {rmse:.2f} kg\n🎯 R² Score: {r2:.4f}\n\n"
        f"🔮 Tomorrow's predicted footprint: {next_pred} kg CO2e")

# Submit today's data
def submit_data():
    try:
        entry = {
            "date": datetime.date.today().isoformat(),
            "car_km": float(car_entry.get() or 0),
            "bus_km": float(bus_entry.get() or 0),
            "bike_km": float(bike_entry.get() or 0),
            "electricity_kwh": float(electricity_entry.get() or 0),
            "meat_meal": int(meat_entry.get() or 0),
            "vegetarian_meal": int(veg_entry.get() or 0)
        }
        save_data(entry)
        total = calculate_footprint(entry)
        messagebox.showinfo("Result", f"✅ Today's carbon footprint: {total} kg CO2e")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers!")

# GUI Setup
initialize_data_file()
root = tk.Tk()
root.title("EcoTrack: Carbon Footprint Tracker")
root.geometry("400x480")

tk.Label(root, text="Enter Today's Activities", font=("Helvetica", 14, "bold")).pack(pady=10)

fields = [
    ("Kilometers by car:", "car_km"),
    ("Kilometers by bus:", "bus_km"),
    ("Kilometers by bike:", "bike_km"),
    ("Electricity used (kWh):", "electricity_kwh"),
    ("Meat meals:", "meat_meal"),
    ("Vegetarian meals:", "vegetarian_meal")
]

entries = {}
for label, key in fields:
    tk.Label(root, text=label).pack()
    entry = tk.Entry(root)
    entry.pack()
    entries[key] = entry

car_entry = entries["car_km"]
bus_entry = entries["bus_km"]
bike_entry = entries["bike_km"]
electricity_entry = entries["electricity_kwh"]
meat_entry = entries["meat_meal"]
veg_entry = entries["vegetarian_meal"]

tk.Button(root, text="Submit", command=submit_data, bg="green", fg="white", width=25).pack(pady=10)
tk.Button(root, text="📊 View Emission Chart & Prediction", command=show_summary, bg="blue", fg="white", width=30).pack(pady=5)
tk.Button(root, text="Exit", command=root.destroy, bg="red", fg="white", width=25).pack(pady=10)

root.mainloop()
