# ----------------------------------------------------
# 3.1 Siapkan fitur dan target regresi
# ----------------------------------------------------

# Pastikan nama kolom di sini benar (pakai 's')
feature_cols_reg = [
    'income',
    'age',
    'subscription_months', # <--- Ini PLURAL
    'num_complaints',
    'has_family_plan',
    'is_premium'
]

X_reg = df[feature_cols_reg]
y_reg = df['monthly_spending']

# ----------------------------------------------------
# 3.2 Train-test split
# ----------------------------------------------------

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# FIX: DEFINISIKAN ULANG PREPROCESSOR DI SINI
# Agar nama kolomnya sinkron dengan X_train_reg
# ----------------------------------------------------

# Sesuaikan list ini dengan tipe data kamu
# Pastikan 'subscription_months' pakai 's'
numerical_cols = ['income', 'age', 'subscription_months', 'num_complaints']
categorical_cols = ['has_family_plan', 'is_premium']

# Buat preprocessor baru
preprocessor = ColumnTransformer(
    transformers=[
        # Gunakan scaler yang sama dengan yang kamu pakai sebelumnya (misal StandardScaler)
        ('num', StandardScaler(), numerical_cols),
        
        # Gunakan encoder yang sama (misal OneHotEncoder)
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# ----------------------------------------------------
# 3.3 Pipeline regresi
# ----------------------------------------------------

reg_model = Pipeline(steps=[
    ('preprocess', preprocessor), # Gunakan preprocessor yang baru diperbaiki
    ('model', LinearRegression())
])

# ----------------------------------------------------
# 3.4 Latih model
# ----------------------------------------------------

reg_model.fit(X_train_reg, y_train_reg)

# ----------------------------------------------------
# 3.5 Prediksi dan evaluasi
# ----------------------------------------------------

y_pred_train = reg_model.predict(X_train_reg)
y_pred_test = reg_model.predict(X_test_reg)

r2_train = r2_score(y_train_reg, y_pred_train)
r2_test = r2_score(y_test_reg, y_pred_test)

mse_train = mean_squared_error(y_train_reg, y_pred_train)
mse_test = mean_squared_error(y_test_reg, y_pred_test)

print(f"R² train : {r2_train:.4f}")
print(f"R² test  : {r2_test:.4f}")
print(f"MSE train: {mse_train:.4f}")
print(f"MSE test : {mse_test:.4f}")

# ----------------------------------------------------
# 3.6 Plot residual
# ----------------------------------------------------
residuals = y_test_reg - y_pred_test

plt.figure(figsize=(6,4))
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(0, linestyle='--', color='red')
plt.xlabel('Predicted Monthly Spending')
plt.ylabel('Residual (y_test - y_pred)')
plt.title('Residual Plot – Linear Regression (Test Set)')
plt.show()