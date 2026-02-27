import streamlit as st
import joblib
import numpy as np
import pandas as pd
from groq import Groq

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Real Estate Machine+", page_icon="🏠", layout="wide")

# --- 2. تحميل الموديلات ---
@st.cache_resource
def load_all_assets():
    try:
        price_model = joblib.load('house_price_model.pkl')
        category_model = joblib.load('house_category_model.pkl')
        scaler = joblib.load('main_scaler.pkl')
        ohe_encoder = joblib.load('encoder.pkl') 
        kmeans = joblib.load('kmeans_model.pkl')
        return price_model, category_model, scaler, ohe_encoder, kmeans
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

price_model, category_model, scaler, ohe_encoder, kmeans = load_all_assets()

# --- 3. إعداد Groq AI ---
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- 4. واجهة المستخدم ---
st.title("🏠 Real Estate Machine+ Valuation")
st.markdown("---")

with st.sidebar:
    st.header("Property Details")
    if ohe_encoder:
        cities_list = list(ohe_encoder.categories_[0])
        months_list = list(ohe_encoder.categories_[1])
        city = st.selectbox("City", options=cities_list)
        month = st.selectbox("Month of Sale", options=months_list)
    else:
        city = st.selectbox("City", options=["Seattle"])
        month = 5

    view = st.slider("View Score", 0, 4, 0)
    condition = st.slider("Condition Score", 1, 5, 3)
    waterfront = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Total Living Area (sqft)", value=2000)
    sqft_above = st.number_input("Sqft Above (Ground Floor)", value=1500)
    sqft_basement = area - sqft_above
    sqft_lot = st.number_input("Sqft Lot", value=5000)
    floors = st.number_input("Number of Floors", value=1.0)

with col2:
    beds = st.number_input("Bedrooms", value=3)
    baths = st.number_input("Bathrooms", value=2.0)
    # 
    yr_built = st.number_input("Year Built", value=1990)
    yr_renovated = st.number_input("Year Renovated (0 if never)", value=0)
    zip_code = st.number_input("Zip Code (Numeric)", value=98101)

if st.button("Run Valuation Machine"):
    if not all([price_model, scaler, ohe_encoder]):
        st.error("Models not loaded properly.")
    else:
        try:
            house_age = 2014 - yr_built
            is_renovated = 1 if yr_renovated > 0 else 0
            years_since_renovation = (2014 - yr_renovated) if is_renovated else house_age
            street_placeholder = 13.0 

            numeric_cols = [
                "bedrooms", "bathrooms", "sqft_living", "sqft_lot", 
                "floors", "sqft_above", "sqft_basement", "house_age", 
                "years_since_renovation", "is_renovated", "street_encoded", 
                "zip_number", "view", "condition", "waterfront"
            ]
            
            numeric_values = [[
                beds, baths, area, sqft_lot, floors, sqft_above, 
                sqft_basement, house_age, years_since_renovation, 
                is_renovated, street_placeholder, zip_code, view, condition, waterfront
            ]]
            
            numeric_scaled = scaler.transform(pd.DataFrame(numeric_values, columns=numeric_cols))
            numeric_scaled_df = pd.DataFrame(numeric_scaled, columns=numeric_cols)

            # معالجة التصنيفات
            cat_df = pd.DataFrame([[city, month]], columns=["city", "month"])
            ohe_features = ohe_encoder.transform(cat_df)
            ohe_cols = ohe_encoder.get_feature_names_out(["city", "month"])
            ohe_df = pd.DataFrame(ohe_features, columns=ohe_cols)

            # الدمج النهائي
            X_combined = pd.concat([numeric_scaled_df, ohe_df], axis=1)
            X_combined['cluster'] = kmeans.predict(numeric_scaled)[0]
            
            expected_order = price_model.feature_names_
            for col in expected_order:
                if col not in X_combined.columns:
                    X_combined[col] = 0 # إضافة العمود بقيمة 0 لو كان ناقص
            
            X_final = X_combined[expected_order]

            # التوقعات
            actual_price = np.exp(price_model.predict(X_final)[0])
            cat_id = category_model.predict(numeric_scaled)[0]
            cat_map = {0: "Budget", 1: "Mid-Range", 2: "Luxury"}
            property_cat = cat_map.get(cat_id, "Standard")

            # --- 6. عرض النتائج ---
            st.markdown("---")
            c1, c2 = st.columns(2)
            price_f = f"${actual_price:,.2f}"
            c1.metric("Predicted Price", price_f)
            c2.metric("Category", property_cat)

            st.markdown("---")
            with st.spinner('Consulting AI Expert...'):
                
                prompt = f"""
            بصفتك خبير عقارات في ولاية واشنطن (King County)،  حلل هذه النتائج وقدم قرار بالشراء ام لا  بناءً على البيانات التالية:
            - المدينة: {city}
            - السعر: {price_f}
            - الفئة: {property_cat}
            - عمر العقار: {house_age} سنة.

            أجب باختصار في 3 نقاط :
            1. هل سعر {price_f} منطقي لعقار في {city} بولاية واشنطن؟
            2. لماذا يعتبر العقار {property_cat} بناءً على مواصفاته وسوق المنطقة؟
            3. نصيحة أخيرة للمشتري (شراء أم انتظار).
            
            ملاحظة: لا تخترع أسماء مدن، التزم بـ {city} فقط.
            """
                
                chat = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.2
                )
                
                # عرض النتيجة بتنسيق يمين لليسار (RTL) مع خلفية مميزة
                st.markdown(f"""
                <div style="direction: rtl; text-align: right; background-color: #e9f7ef; padding: 20px; border-radius: 12px; border: 1px solid #28a745;">
                    <h3 style="color: #155724; margin-top: 0;">🤖 رأي الخبير العقاري (AI):</h3>
                    {chat.choices[0].message.content}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Real Estate Machine+ v2.0")
st.markdown("Developed and Optimized by **Osman**")


