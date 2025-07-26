import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set up the app with wide layout
st.set_page_config(page_title="Inventory Risk Prediction", layout="wide", page_icon="📊")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
        }

        /* Sidebar styling */
        .css-1aumxhk {
            background-color: #343a40 !important;
            color: white !important;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
        }
        .sidebar .sidebar-content .block-container {
            color: white;
        }

        /* Title styling */
        .css-10trblm {
            color: #2c3e50;
            font-weight: 700;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }

        /* Header styling */
        h1, h2, h3 {
            color: #ffffff !important;
        }

        /* Button styling */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 24px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* Input field styling */
        .stNumberInput, .stSelectbox, .stTextInput {
            margin-bottom: 15px;
        }

        /* Card styling for predictions */
        .prediction-card {
            padding: 25px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .prediction-card:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .high-risk {
            background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
            color: white;
            border-left: 5px solid #e74c3c;
        }
        .medium-risk {
            background: linear-gradient(135deg, #ffd166, #ffe066);
            color: #2c3e50;
            border-left: 5px solid #f39c12;
        }
        .low-risk {
            background: linear-gradient(135deg, #06d6a0, #48cae4);
            color: white;
            border-left: 5px solid #2ecc71;
        }

        /* Metric cards styling */
        .metric-card {
            padding: 20px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 20px;
        }
        .metric-title {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #2c3e50;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 25px;
            background-color: #ecf0f1;
            border-radius: 5px 5px 0 0 !important;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3498db !important;
            color: white !important;
        }

        /* Dataframe styling */
        .dataframe {
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Custom columns spacing */
        .stColumn {
            padding: 0 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App title with custom styling
st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 30px;">
        <h1 style="margin: 0; color: #ffffff;">📊 Inventory Risk Prediction System</h1>
        <span style="margin-left: auto; font-size: 14px; color: #7f8c8d;"></span>
    </div>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"stock_risk_prediction_dataset.csv")

        # Preprocessing
        categorical_cols = ['season', 'item_category', 'supplier_reliability', 'stock_status']
        for col in categorical_cols:
            df[col] = df[col].str.lower().str.strip()

        # Calculate additional features
        df['forecast_error'] = df['avg_daily_demand'] * 30 - df['sales_last_30_days']
        df['stock_turnover_ratio'] = df['sales_last_30_days'] / df['current_stock'].replace(0, 1)

        df = df.drop_duplicates()

        # Ensure required columns exist
        required_columns = [
            'current_stock', 'avg_daily_demand', 'lead_time_days', 'reorder_point',
            'sales_last_30_days', 'stock_turnover_ratio', 'forecast_error',
            'season', 'item_category', 'supplier_reliability', 'stock_status'
        ]

        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return None

        return df

    except Exception as e:
        st.error(f"Error loading data file: {str(e)}")
        return None


df = load_data()

if df is None:
    st.stop()

# Sidebar for navigation with custom styling
with st.sidebar:
    st.markdown("""
        <div style="padding: 15px; background-color: #3498db; color: white; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">Navigation</h2>
        </div>
    """, unsafe_allow_html=True)

    options = st.radio("Select Page:",
                       ["📊 Data Overview",
                        "🔮 Risk Prediction",
                        "📈 Inventory Analysis",
                        "📝 Model Performance"],
                       label_visibility="collapsed")

# Data Overview Page
if options == "📊 Data Overview":
    st.header("Dataset Overview")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Statistics", "📊 Visualizations"])

    with tab1:
        st.subheader("First 10 Rows of Data")
        st.dataframe(df.head(10), height=350)

        st.subheader("Data Information")
        st.write(f"Total Rows: {df.shape[0]}")
        st.write(f"Total Columns: {df.shape[1]}")

    with tab2:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

        st.subheader("Missing Values")
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        st.dataframe(missing_data)

    with tab3:
        st.subheader("Stock Status Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='stock_status', data=df,hue = 'stock_status',palette="viridis", ax=ax,legend = False)
        ax.set_title("Distribution of Risk Levels", fontsize=14)
        ax.set_xlabel("Risk Level", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)

        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if not numeric_cols.empty:
            selected_feature = st.selectbox("Select feature to visualize:", numeric_cols)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[selected_feature], kde=True, color="#3498db", ax=ax)
            ax.set_title(f"Distribution of {selected_feature}", fontsize=14)
            ax.set_xlabel(selected_feature, fontsize=12)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for visualization")

# Risk Prediction Page
elif options == "🔮 Risk Prediction":
    st.header("Stock Risk Prediction")

    # Check if model exists, otherwise train it
    model_path = 'stock_risk_model.joblib'
    if not os.path.exists(model_path):
        with st.spinner("Training model... This may take a few moments."):
            # Preprocess data
            df_model = df.copy()
            df_model = pd.get_dummies(df_model, columns=['season', 'item_category', 'supplier_reliability'])

            X = df_model.drop('stock_status', axis=1)
            y = df_model['stock_status']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Save model
            joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    # Prediction form in a container
    with st.container():
        st.subheader("Predict Stock Risk for New Item")
        st.markdown("Enter the item details below to assess its inventory risk level.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Inventory Details")
            current_stock = st.number_input("Current Stock (units)", min_value=0, help="Current inventory level")
            avg_daily_demand = st.number_input("Average Daily Demand (units/day)", min_value=0.0, format="%.2f",
                                               help="Expected daily sales")
            lead_time_days = st.number_input("Lead Time (days)", min_value=1, help="Time to receive new stock")
            reorder_point = st.number_input("Reorder Point (units)", min_value=0,
                                            help="Inventory level triggering reorder")
            sales_last_30_days = st.number_input("Sales Last 30 Days (units)", min_value=0,
                                                 help="Actual sales in past month")

        with col2:
            st.markdown("#### Categorical Features")
            season = st.selectbox("Season", df['season'].unique(), help="Current season affecting demand")
            item_category = st.selectbox("Item Category", df['item_category'].unique(), help="Product category")
            supplier_reliability = st.selectbox("Supplier Reliability", df['supplier_reliability'].unique(),
                                                help="Supplier performance rating")

            # Calculate derived features
            forecast_error = avg_daily_demand * 30 - sales_last_30_days
            stock_turnover_ratio = sales_last_30_days / (current_stock if current_stock > 0 else 1)

            st.markdown("#### Calculated Metrics")
            st.metric("Forecast Error", f"{forecast_error:.2f}")
            st.metric("Stock Turnover Ratio", f"{stock_turnover_ratio:.2f}")

    if st.button("Predict Stock Risk", key="predict_button"):
        # Prepare input data
        input_data = pd.DataFrame({
            'current_stock': [current_stock],
            'avg_daily_demand': [avg_daily_demand],
            'lead_time_days': [lead_time_days],
            'reorder_point': [reorder_point],
            'sales_last_30_days': [sales_last_30_days],
            'stock_turnover_ratio': [stock_turnover_ratio],
            'forecast_error': [forecast_error]
        })

        # Add one-hot encoded categorical features
        for cat_col in ['season', 'item_category', 'supplier_reliability']:
            for val in df[cat_col].unique():
                col_name = f"{cat_col}_{val}"
                input_data[col_name] = [1 if (cat_col == 'season' and val == season) or
                                             (cat_col == 'item_category' and val == item_category) or
                                             (cat_col == 'supplier_reliability' and val == supplier_reliability) else 0]

        # Ensure we have all columns the model expects
        model_columns = model.feature_names_in_
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Add missing columns with default value 0
        input_data = input_data[model_columns]  # Reorder columns to match model

        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            # Get probability for the predicted class
            pred_prob = proba[list(model.classes_).index(prediction)] * 100

            # Display results in a styled card
            st.markdown("## Prediction Result")

            if prediction == 'high risk':
                st.markdown(f"""
                    <div class="prediction-card high-risk">
                        <h2 style="color: white;">🚨 High Risk</h2>
                        <p style="font-size: 18px;">Probability: {pred_prob:.1f}%</p>
                        <h3 style="color: white;">Recommended Actions:</h3>
                        <ul style="color: white;">
                            <li>Immediate action required</li>
                            <li>Consider emergency replenishment</li>
                            <li>Analyze demand forecasting accuracy</li>
                            <li>Review supplier contracts</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif prediction == 'medium risk':
                st.markdown(f"""
                    <div class="prediction-card medium-risk">
                        <h2>⚠️ Medium Risk</h2>
                        <p style="font-size: 18px;">Probability: {pred_prob:.1f}%</p>
                        <h3>Recommended Actions:</h3>
                        <ul>
                            <li>Review supplier lead times</li>
                            <li>Consider safety stock adjustments</li>
                            <li>Monitor sales trends closely</li>
                            <li>Plan for potential stockouts</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-card low-risk">
                        <h2 style="color: white;">✅ Low Risk</h2>
                        <p style="font-size: 18px; color: white;">Probability: {pred_prob:.1f}%</p>
                        <h3 style="color: white;">Recommended Actions:</h3>
                        <ul style="color: white;">
                            <li>Maintain current inventory levels</li>
                            <li>Continue monitoring demand patterns</li>
                            <li>Consider promotional opportunities</li>
                            <li>Review for potential overstock</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            # Show probability distribution
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Risk Level': model.classes_,
                'Probability': proba
            }).sort_values('Probability', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Probability', y='Risk Level', data=prob_df, palette="viridis", ax=ax)
            ax.set_title("Prediction Confidence", fontsize=14)
            ax.set_xlabel("Probability", fontsize=12)
            ax.set_ylabel("Risk Level", fontsize=12)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Inventory Analysis Page
elif options == "📈 Inventory Analysis":
    st.header("Inventory Analysis Dashboard")

    # Filter section in an expander
    with st.expander("🔍 Filter Inventory", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Inventory Levels")
            min_stock = st.number_input("Minimum Current Stock",
                                        min_value=0,
                                        value=int(df['current_stock'].min()))
            max_stock = st.number_input("Maximum Current Stock",
                                        min_value=0,
                                        value=int(df['current_stock'].max()))

            st.markdown("#### Risk Status")
            risk_status = st.multiselect("Select Risk Levels",
                                         options=df['stock_status'].unique(),
                                         default=df['stock_status'].unique())

        with col2:
            st.markdown("#### Product Characteristics")
            item_cat = st.multiselect("Item Categories",
                                      options=df['item_category'].unique(),
                                      default=df['item_category'].unique())
            supplier_rel = st.multiselect("Supplier Reliability",
                                          options=df['supplier_reliability'].unique(),
                                          default=df['supplier_reliability'].unique())

    # Apply filters
    filtered_df = df[
        (df['current_stock'] >= min_stock) &
        (df['current_stock'] <= max_stock) &
        (df['stock_status'].isin(risk_status)) &
        (df['item_category'].isin(item_cat)) &
        (df['supplier_reliability'].isin(supplier_rel))
        ]

    # Summary metrics in columns
    st.markdown(f"### Found {len(filtered_df)} items matching your criteria")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">High Risk Items</div>
                <div class="metric-value" style="color: #e74c3c;">{}</div>
            </div>
        """.format(len(filtered_df[filtered_df['stock_status'] == 'high risk'])), unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Medium Risk Items</div>
                <div class="metric-value" style="color: #f39c12;">{}</div>
            </div>
        """.format(len(filtered_df[filtered_df['stock_status'] == 'medium risk'])), unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Low Risk Items</div>
                <div class="metric-value" style="color: #2ecc71;">{}</div>
            </div>
        """.format(len(filtered_df[filtered_df['stock_status'] == 'low risk'])), unsafe_allow_html=True)

    # Tabs for different views
    tab1, tab2 = st.tabs(["📋 Filtered Data", "📊 Visual Analysis"])

    with tab1:
        st.dataframe(filtered_df, height=500)

        # Download button for filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name='filtered_inventory.csv',
            mime='text/csv'
        )

    with tab2:
        st.subheader("Inventory Risk Distribution")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Countplot
        sns.countplot(x='stock_status', data=filtered_df, palette="viridis", ax=ax1)
        ax1.set_title("Risk Status Count", fontsize=14)
        ax1.set_xlabel("Risk Level", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)

        # Boxplot
        sns.boxplot(x='stock_status', y='current_stock', data=filtered_df, palette="viridis", ax=ax2)
        ax2.set_title("Stock Levels by Risk Status", fontsize=14)
        ax2.set_xlabel("Risk Level", fontsize=12)
        ax2.set_ylabel("Current Stock", fontsize=12)

        st.pyplot(fig)

        # Additional visualization
        st.subheader("Lead Time vs. Current Stock")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x='lead_time_days', y='current_stock', hue='stock_status',
                        data=filtered_df, palette="viridis", ax=ax, s=100)
        ax.set_title("Lead Time vs. Current Stock by Risk Level", fontsize=14)
        ax.set_xlabel("Lead Time (days)", fontsize=12)
        ax.set_ylabel("Current Stock", fontsize=12)
        st.pyplot(fig)

# Model Performance Page
elif options == "📝 Model Performance":
    st.header("Model Performance Evaluation")

    # Preprocess data
    with st.spinner("Preparing data for model evaluation..."):
        df_model = df.copy()
        df_model = pd.get_dummies(df_model, columns=['season', 'item_category', 'supplier_reliability'])

        X = df_model.drop('stock_status', axis=1)
        y = df_model['stock_status']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Check if model exists, otherwise train it
        model_path = 'stock_risk_model.joblib'
        if not os.path.exists(model_path):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
        else:
            model = joblib.load(model_path)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Performance metrics in columns
    st.markdown("### Model Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">{accuracy:.2%}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        precision = report['weighted avg']['precision']
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value">{precision:.2%}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        recall = report['weighted avg']['recall']
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Recall</div>
                <div class="metric-value">{recall:.2%}</div>
            </div>
        """, unsafe_allow_html=True)

    # Tabs for different evaluation views
    tab1, tab2, tab3 = st.tabs(["📝 Classification Report", "📊 Confusion Matrix", "🔍 Feature Importance"])

    with tab1:
        st.subheader("Detailed Classification Metrics")
        st.dataframe(report_df.style.format("{:.2f}"), height=400)

    with tab2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                    xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)
        st.pyplot(fig)

    with tab3:
        st.subheader("Feature Importance")
        feature_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_imp.head(10), palette="viridis", ax=ax)
        ax.set_title("Top 10 Most Important Features", fontsize=14)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        st.pyplot(fig)

        # Show full feature importance table
        st.subheader("Complete Feature Importance Table")
        st.dataframe(feature_imp, height=500)