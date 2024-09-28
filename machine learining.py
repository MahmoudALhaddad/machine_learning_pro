import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from fstring import fstring

class RegressionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Regression Analysis Tool")

        # Load Data Frame
        self.create_load_frame()

        # Regression Models Frame
        self.create_model_frame()

        # Results Frame
        self.create_results_frame()

        self.dataset = None
        self.pipeline = None

    def create_load_frame(self):
        load_frame = ttk.LabelFrame(self.master, text="Load Dataset")
        load_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.load_button = ttk.Button(load_frame, text="Browse...", command=self.load_dataset)
        self.load_button.pack(pady=10)

        predict_button = ttk.Button(load_frame, text="Open Predictor", command=self.create_predictor_window)
        predict_button.pack(pady=(0, 10))

    def create_predictor_window(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        # Predictor Window
        predictor_window = tk.Toplevel(self.master)
        predictor_window.title("Predictor")

        ttk.Label(predictor_window, text="Choose Regression Algorithm:").grid(
            row=0, column=0, columnspan=2, padx=10, pady=10
        )

        self.predictor_model_var = tk.StringVar(
            predictor_window, value="Multiple Linear Regression"
        )  # Set default value
        ttk.Radiobutton(
            predictor_window,
            text="Multiple Linear Regression",
            variable=self.predictor_model_var,
            value="Multiple Linear Regression",
        ).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(
            predictor_window,
            text="Random Forest Regression",
            variable=self.predictor_model_var,
            value="Random Forest Regression",
        ).grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(
            predictor_window,
            text="Gradient Boosting Regression",
            variable=self.predictor_model_var,
            value="Gradient Boosting Regression",
        ).grid(row=3, column=0, sticky="w")

        ttk.Label(predictor_window, text="Enter new data point:").grid(
            row=4, column=0, columnspan=2, padx=10, pady=10
        )

        # Dynamically create input fields (Entry widgets) for all features
        self.input_fields = {}
        for i, feature in enumerate(self.X.columns):
            ttk.Label(predictor_window, text=feature).grid(
                row=i + 5, column=0, padx=5, pady=5, sticky="w"
            )

            # Create Entry widget for all features (categorical and numerical)
            entry = ttk.Entry(predictor_window)
            entry.grid(row=i + 5, column=1, padx=5, pady=5)
            self.input_fields[feature] = entry

        # Prediction Button
        predict_button = ttk.Button(
            predictor_window, text="Predict", command=self.make_prediction
        )
        predict_button.grid(
            row=len(self.X.columns) + 5, column=0, columnspan=2, padx=10, pady=10
        )

        # Result Label
        self.prediction_result = ttk.Label(predictor_window, text="")
        self.prediction_result.grid(
            row=len(self.X.columns) + 6, column=0, columnspan=2, padx=10, pady=10
        )



    def make_prediction(self):
        try:
            # Retrain the model with the selected algorithm before making predictions
            model_name = self.predictor_model_var.get()
            self.preprocess_data(model_name)  # This sets self.categorical_features

            # Get input data and convert to a DataFrame
            new_data = pd.DataFrame(
                {feature: [value.get()] for feature, value in self.input_fields.items()}
            )

            # Validate categorical feature inputs and encode them if valid
            label_encoders = {}  # Store label encoders for each categorical feature
            for feature in self.categorical_features:
                valid_categories = self.dataset[feature].unique()
                entered_category = new_data[feature][0]
                
                if entered_category not in valid_categories:
                    messagebox.showerror(
                        "Error",
                        f"Invalid category '{entered_category}' entered for feature '{feature}'. \
                        Valid categories are: {', '.join(valid_categories)}",
                    )
                    return
                
                # Create and fit a LabelEncoder for each categorical feature
                le = LabelEncoder()
                le.fit(valid_categories)  
                label_encoders[feature] = le

                # Transform the new_data using the fitted LabelEncoder
                new_data[feature] = le.transform(new_data[feature])


            # Check for NaNs in the input data
            if new_data.isnull().values.any():
                messagebox.showerror(
                    "Error", "Input data contains missing values. Please fill in all fields."
                )
                return

            # Predict
            prediction = self.pipeline.predict(new_data)[0]
            self.prediction_result.config(
                text=f"Predicted {self.target_variable}: {prediction:.2f}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {e}")


    def create_model_frame(self):
        model_frame = ttk.LabelFrame(self.master, text="Regression Models")
        model_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(model_frame, text="Choose a model:").pack(pady=(10, 0))

        self.model_var = tk.StringVar(value="Multiple Linear Regression")

        ttk.Radiobutton(model_frame, text="Multiple Linear Regression", variable=self.model_var, value="Multiple Linear Regression").pack(anchor="w")
        ttk.Radiobutton(model_frame, text="Random Forest Regression", variable=self.model_var, value="Random Forest Regression").pack(anchor="w")
        ttk.Radiobutton(model_frame, text="Gradient Boosting Regression", variable=self.model_var, value="Gradient Boosting Regression").pack(anchor="w")

        self.run_button = ttk.Button(model_frame, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack(pady=10)


    def preprocess_data(self, model_name):
        # Automatically detect target variable (last column)
        self.target_variable = self.dataset.columns[-1]

        # Drop rows with NaN values in the target variable
        self.dataset.dropna(subset=[self.target_variable], inplace=True)

        self.X = self.dataset.drop(self.target_variable, axis=1)
        self.y = self.dataset[self.target_variable]

        # Automatically detect categorical and numerical features
        self.categorical_features = self.X.select_dtypes(include=['object']).columns
        self.numeric_features = self.X.select_dtypes(exclude=['object']).columns

        # Apply LabelEncoder to categorical features
        le = LabelEncoder()
        for feature in self.categorical_features:
            self.X[feature] = le.fit_transform(self.X[feature])

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Create pipelines for numerical and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Create the regression model pipeline
        if model_name == "Multiple Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest Regression":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "Gradient Boosting Regression":
            model = GradientBoostingRegressor(random_state=42)
        else:
            raise ValueError(fstring("Unknown model name: {model_name}"))

        # Combine preprocessor and model into a single pipeline
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', model)])

        # Fit the pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)
        
        # Make predictions on the test set
        y_pred = self.pipeline.predict(self.X_test)

        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred) 

        return mse, r2

    def create_results_frame(self):
        results_frame = ttk.LabelFrame(self.master, text="Results")
        results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=60, height=10)
        self.results_text.pack(pady=10)

        # Plot 
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().pack()

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
                self.update_results("Dataset loaded", "", "")  # Clear previous results
            except Exception as e:
                messagebox.showerror("Error", "Error loading dataset: " + str(e))

    def run_analysis(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        try:
            model_name = self.model_var.get()
            mse, r2 = self.preprocess_data(model_name)  
            self.update_results(model_name, r2, mse, plot=True) 
        except Exception as e:
            messagebox.showerror("Error", "Error running analysis: " + str(e)) 


    def update_results(self, model_name, r2, mse, plot=False):
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Model: " + model_name + "\nR-squared: " + str(r2) + "\nMean Squared Error: " + str(mse) + "\n") 
        
        # Only plot if plot flag is True
        if plot:
            self.plot_results(model_name)


    def plot_results(self, model_name):
        self.ax.clear()

        # Select the relevant column for plotting based on the model name
        if model_name == "Multiple Linear Regression":
            default_col = 'Years of Experience'  # Default for multiple linear regression
        elif model_name == "Random Forest Regression":
            # For Random Forest, let's just use the first available numerical feature
            default_col = next((col for col in self.X_test.columns if self.X_test[col].dtype in ['int64', 'float64']), None)
        else:  # Gradient Boosting Regression (or default)
            default_col = 'Years of Experience'  # Default for gradient boosting regression

        # Get a list of available features (excluding the target)
        available_features = [col for col in self.X.columns if self.X[col].dtype in ['int64', 'float64']]

        if available_features: # Check if there are any numerical columns available

            # Input Dialog with Dropdown List
            def get_input():
                col_to_plot = self.feature_var.get()
                self.plot_with_feature(col_to_plot, model_name)

            # Create input dialog box
            input_dialog = tk.Toplevel(self.master)
            input_dialog.title("Choose Feature")
            ttk.Label(input_dialog, text="Select a feature to plot against the target:").pack(pady=10)

            self.feature_var = tk.StringVar(input_dialog)
            self.feature_var.set(default_col)  # Default selection

            feature_dropdown = ttk.Combobox(input_dialog, textvariable=self.feature_var, values=available_features)
            feature_dropdown.pack()

            ttk.Button(input_dialog, text="Plot", command=get_input).pack(pady=10)
        else:
            messagebox.showwarning("Warning", fstring("No valid numerical columns found in dataset for plotting."))

    def plot_with_feature(self, col_to_plot, model_name):
        self.ax.cla()
        X_test_plot = self.X_test[col_to_plot]
        if isinstance(X_test_plot, pd.DataFrame):
            X_test_plot = X_test_plot.iloc[:, 0]

        # Handle case where col_to_plot is not in X_train (e.g., one-hot encoded)
        if col_to_plot not in self.X_train.columns:
            # Find the original categorical feature if col_to_plot is one-hot encoded
            original_feature = next(
                (
                    feature
                    for feature in self.categorical_features
                    if col_to_plot.startswith(feature + "_")
                ),
                None,
            )

            if original_feature:
                # Apply LabelEncoder to get numerical values for plotting
                X_train_plot = (
                    LabelEncoder()
                    .fit_transform(self.X_train[original_feature])
                    .astype(float)
                )
            else:
                messagebox.showwarning(
                    "Warning",
                    f"The selected feature '{col_to_plot}' is not suitable for plotting.",
                )
                return
        else:
            # Handle case where col_to_plot is already in X_train
            X_train_plot = self.X_train[col_to_plot]
            if isinstance(X_train_plot, pd.DataFrame):
                X_train_plot = X_train_plot.iloc[:, 0]

        y_pred = self.pipeline.predict(self.X_test)
        y_train_pred = self.pipeline.predict(self.X_train)

        # Plot both train and test data
        self.ax.scatter(X_train_plot, self.y_train, color="green", label="Actual (Train)")
        self.ax.plot(
            X_train_plot, y_train_pred, color="lightgreen", label="Predicted (Train)"
        )
        self.ax.scatter(X_test_plot, self.y_test, color="blue", label="Actual (Test)")
        self.ax.plot(X_test_plot, y_pred, color="red", label="Predicted (Test)")
        self.ax.set_xlabel(col_to_plot)
        self.ax.set_ylabel(self.target_variable)
        self.ax.legend()
        self.ax.set_title(f"Truth or Bluff ({self.model_var.get()})")
        self.canvas.draw()
  


def main():
    root = tk.Tk()
    app = RegressionGUI(root)
    root.resizable(False, False)
    root.mainloop()

if __name__ == "__main__":
    main()
