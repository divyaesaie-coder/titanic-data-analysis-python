import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style for better graphs
plt.style.use("ggplot")

# Load dataset
df = pd.read_csv("data.csv")

# =========================
# BASIC DATA ANALYSIS
# =========================

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== TOTAL ROWS AND COLUMNS ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# =========================
# DATA CLEANING
# =========================

# Fill missing age values with average age
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("\n========== AFTER CLEANING ==========")
print(df.isnull().sum())

# =========================
# INSIGHTS
# =========================

# Average age
average_age = df["Age"].mean()
print("\nAverage Age:", average_age)

# Survival rate
survival_rate = df["Survived"].mean() * 100
print("Survival Rate: {:.2f}%".format(survival_rate))

# Average fare
average_fare = df["Fare"].mean()
print("Average Fare:", average_fare)

# =========================
# VISUALIZATIONS
# =========================

# 1. Gender Distribution Bar Chart
df["Sex"].value_counts().plot(kind="bar")

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

# 2. Passenger Class Distribution
df["Pclass"].value_counts().plot(kind="bar")

plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Count")

plt.show()

# 3. Scatter Plot - Age vs Fare
plt.scatter(df["Age"], df["Fare"])

plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.show()

# 4. Heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# 5. Survival Percentage Pie Chart
df["Survived"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Survival Percentage")
plt.ylabel("")

plt.show()

# 6. Age Distribution Histogram
plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()

# 7. Survival Based on Gender
sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival Based on Gender")

plt.show()

# =========================
# CORRELATION ANALYSIS
# =========================

print("\n========== CORRELATION ANALYSIS ==========")
print(df.corr(numeric_only=True))

# =========================
# FINAL OBSERVATIONS
# =========================

print("\n========== OBSERVATIONS ==========")

print("1. Female passengers had higher survival rates.")
print("2. Most passengers belonged to 3rd class.")
print("3. Passenger fares varied significantly.")
print("4. Younger passengers were more common.")
print("5. Passenger class affected survival chances.")

print("\nProject Completed Successfully!")
