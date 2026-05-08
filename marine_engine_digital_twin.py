import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ========================= إعداد الصفحة =========================
st.set_page_config(
    page_title="النموذج الرقمي التوأم للمحرك البحري",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= نظام الذكاء الاصطناعي =========================
class AdvancedMarineAI:
    def __init__(self):
        self.rpm_model = RandomForestRegressor(n_estimators=100)
        self.fuel_model = RandomForestRegressor(n_estimators=100)
        self.temp_model = RandomForestRegressor(n_estimators=100)
        self.vib_model = RandomForestRegressor(n_estimators=100)
        self.rul_model = RandomForestRegressor(n_estimators=100)
        self.fault_predictor = RandomForestRegressor(n_estimators=50)
        self.efficiency_model = RandomForestRegressor(n_estimators=50)
        
        np.random.seed(42)
        X_train = np.random.rand(500, 4)
        X_train[:,0] = X_train[:,0] * 100
        X_train[:,1] = X_train[:,1] * 100
        X_train[:,2] = X_train[:,2] * 50
        X_train[:,3] = X_train[:,3] * 2000
        
        y_rpm = 800 + X_train[:,0]*5 - X_train[:,1]*2 + X_train[:,2]*0.5 + np.random.randn(500)*20
        y_fuel = 0.1*X_train[:,0] + 0.02*X_train[:,1] + 0.01*X_train[:,2] + np.random.randn(500)*0.05
        y_temp = 60 + 0.3*X_train[:,0] + 0.2*X_train[:,1] + np.random.randn(500)*3
        y_vib = 0.5 + 0.02*X_train[:,0] + 0.03*X_train[:,1] + np.random.randn(500)*0.1
        y_rul = 500 - 2*X_train[:,1] - 0.1*X_train[:,3] + np.random.randn(500)*10
        y_fault = (X_train[:,1] > 70) | (y_vib > 3.0) | (y_temp > 100)
        y_efficiency = 85 - X_train[:,0]*0.3 - X_train[:,1]*0.4 + np.random.randn(500)*3
        
        self.rpm_model.fit(X_train, y_rpm)
        self.fuel_model.fit(X_train, y_fuel)
        self.temp_model.fit(X_train, y_temp)
        self.vib_model.fit(X_train, y_vib)
        self.rul_model.fit(X_train, y_rul)
        self.fault_predictor.fit(X_train, y_fault)
        self.efficiency_model.fit(X_train, y_efficiency)
        
    def predict_all(self, load, wear, water_temp, hours):
        X = np.array([[load, wear, water_temp, hours]])
        rpm = self.rpm_model.predict(X)[0]
        fuel = self.fuel_model.predict(X)[0]
        temp = self.temp_model.predict(X)[0]
        vib = self.vib_model.predict(X)[0]
        rul = max(0, self.rul_model.predict(X)[0])
        fault_prob = self.fault_predictor.predict(X)[0]
        efficiency = max(0, min(100, self.efficiency_model.predict(X)[0]))
        return rpm, fuel, temp, vib, rul, fault_prob, efficiency

# ========================= التخزين المؤقت للبيانات =========================
if 'ai' not in st.session_state:
    st.session_state.ai = AdvancedMarineAI()
if 'history' not in st.session_state:
    st.session_state.history = []

# ========================= الشريط الجانبي =========================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/ship.png", width=80)
    st.title("🏛️ جامعة شرق بورسعيد التكنولوجية")
    st.markdown("**قسم تشغيل وصيانة السفن**")
    st.markdown("---")
    st.markdown("### 👨‍🏫 تحت إشراف")
    st.markdown("**الأستاذ الدكتور / حسين المصري**")
    st.markdown("---")
    st.markdown("### 👥 فريق العمل")
    st.markdown("- حازم محمد بسيوني عبد المقصود")
    st.markdown("- أحمد حمدي فتوح السعودي")
    st.markdown("- يوسف محمد عبد الحميد احمد")
    st.markdown("- أحمد رضا السيد أحمد عبد الستار")
    st.markdown("- عمرو محمود إسحق صالح")
    st.markdown("- أحمد محمد أحمد محمود السيد")
    st.markdown("- السيد مسعد محمد")
    st.markdown("---")

    st.header("📊 بيانات التشغيل")
    load = st.slider("⚡ نسبة الحمل (%)", 0, 100, 65)
    wear = st.slider("🔧 تآكل المحور (%)", 0, 100, 25)
    water_temp = st.slider("🌊 حرارة ماء البحر (°C)", 0, 50, 25)
    hours = st.number_input("⏱️ ساعات التشغيل (h)", min_value=0, max_value=5000, value=1200, step=100)
    
    predict_button = st.button("🚀 تنبؤ فوري وتحليل ذكي", use_container_width=True)
    auto_refresh = st.checkbox("🔄 تحديث تلقائي (كل 3 ثوانٍ)", value=False)

# ========================= العنوان الرئيسي =========================
st.title("⚙️ النموذج الرقمي التوأم للمحرك البحري")
st.markdown("### 💎 النظام الذكي للصيانة التنبؤية – الجيل الرابع")
st.markdown("---")

# ========================= التنبؤ =========================
if predict_button or auto_refresh:
    rpm, fuel, oil_temp, vib, rul, fault_prob, efficiency = st.session_state.ai.predict_all(load, wear, water_temp, hours)
    
    # تحديث السجل
    st.session_state.history.append({
        'timestamp': datetime.now(),
        'rpm': rpm, 'fuel': fuel, 'oil_temp': oil_temp, 'vib': vib, 'rul': rul,
        'load': load, 'wear': wear
    })
    if len(st.session_state.history) > 60:
        st.session_state.history = st.session_state.history[-60:]
    
    # ========================= عرض النتائج =========================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔄 RPM", f"{rpm:.0f}", delta=None)
        st.metric("⛽ استهلاك الوقود", f"{fuel:.2f} t/h")
    with col2:
        st.metric("🌡️ حرارة الزيت", f"{oil_temp:.1f} °C", 
                  delta="مرتفعة" if oil_temp > 95 else "طبيعية")
        st.metric("📳 الاهتزاز", f"{vib:.2f} m/s²")
    with col3:
        st.metric("🔧 العمر المتبقي", f"{rul:.0f} ساعة")
        st.metric("⚠️ احتمالية العطل", f"{fault_prob*100:.1f} %")
    with col4:
        st.metric("⚡ كفاءة الطاقة", f"{efficiency:.0f} %")
        if efficiency < 50:
            st.error("🔴 كفاءة منخفضة جداً")
        elif efficiency < 70:
            st.warning("🟡 كفاءة متوسطة")
        else:
            st.success("🟢 كفاءة ممتازة")
    
    # ========================= الألوان التحذيرية =========================
    if oil_temp > 100:
        st.error("🚨 تحذير: حرارة الزيت مرتفعة جداً >100°C!")
    elif oil_temp > 85:
        st.warning("⚠️ تنبيه: حرارة الزيت مرتفعة")
    
    if fault_prob > 0.7:
        st.error("🚨 خطر: احتمالية عطل وشيك تزيد عن 70%!")
    elif fault_prob > 0.4:
        st.warning("⚠️ تنبيه: احتمالية عطل متوسطة")
    
    # ========================= الأنظمة الذكية =========================
    st.markdown("---")
    st.subheader("🧠 الأنظمة الذكية المتقدمة")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 تحسين الوقود", "🔮 تنبؤ الأعطال", "📅 جدولة الصيانة", "📊 تحليل الاتجاهات", "🔄 مقارنة مع جديد"
    ])
    
    with tab1:
        best_rpm = 800 + load * 2
        savings = fuel * 0.15
        st.info(f"""
        **📉 تحسين استهلاك الوقود:**
        - أفضل RPM لتقليل الاستهلاك: **{best_rpm:.0f}**
        - التوفير المتوقع: **{savings:.2f} t/h** (~15%)
        - نصيحة: حافظ على RPM بين 800-1000
        """)
    
    with tab2:
        if vib > 3.0 or oil_temp > 95 or fault_prob > 0.6:
            st.error(f"""
            **🔴 تحذير عالي الخطورة!**
            - احتمالية العطل: **{fault_prob*100:.1f}%**
            - متوقع خلال: **10-15 دقيقة**
            - الإجراء: خفض الحمل فوراً إلى 50%
            """)
        elif vib > 2.0 or oil_temp > 85 or fault_prob > 0.3:
            st.warning(f"""
            **🟡 تنبيه متوسط الخطورة**
            - احتمالية العطل: **{fault_prob*100:.1f}%**
            - متوقع خلال: **30-60 دقيقة**
            """)
        else:
            st.success(f"🟢 الحالة مستقرة – احتمالية العطل {fault_prob*100:.1f}%")
    
    with tab3:
        if rul < 100:
            schedule = "خلال 24 ساعة (صيانة عاجلة)"
        elif rul < 200:
            schedule = "خلال هذا الأسبوع"
        elif rul < 350:
            schedule = "خلال أسبوعين"
        else:
            schedule = "بعد شهر"
        st.info(f"""
        **📅 جدولة الصيانة الذكية:**
        - العمر المتبقي: **{rul:.0f}** ساعة
        - الصيانة الموصى بها: **{schedule}**
        """)
    
    with tab4:
        future_temp = oil_temp + (load * 0.1)
        future_vib = vib + (wear * 0.02)
        st.write("**📊 توقع بعد ساعة من التشغيل:**")
        cola, colb = st.columns(2)
        cola.metric("حرارة الزيت", f"{oil_temp:.1f}°C", f"{future_temp - oil_temp:+.1f}°C")
        colb.metric("الاهتزاز", f"{vib:.2f}", f"{future_vib - vib:+.2f}")
    
    with tab5:
        new_fuel = 0.08 * load + 0.5
        new_eff = 85 - load * 0.2
        fuel_loss = (fuel - new_fuel) / new_fuel * 100 if new_fuel > 0 else 0
        st.write("**🔄 المقارنة مع محرك جديد:**")
        cola, colb = st.columns(2)
        cola.metric("استهلاك الوقود (حالياً)", f"{fuel:.2f} t/h", f"{fuel_loss:.1f}% worse" if fuel_loss > 0 else None)
        colb.metric("كفاءة الطاقة (حالياً)", f"{efficiency:.0f}%", f"{efficiency - new_eff:.0f}%")
    
    # ========================= علامات التبويب =========================
    st.markdown("---")
    tab_graph, tab_3d = st.tabs(["📈 تطور المتغيرات", "🎬 النموذج 3D الذكي"])
    
    with tab_graph:
        if len(st.session_state.history) > 1:
            df = pd.DataFrame(st.session_state.history)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rpm'], mode='lines', name='RPM', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['fuel']*100, mode='lines', name='الوقود (×100)', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['oil_temp'], mode='lines', name='حرارة الزيت', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vib']*2, mode='lines', name='الاهتزاز (×2)', line=dict(color='blue')))
            fig.update_layout(title="تطور المتغيرات الزمنية", xaxis_title="الوقت", yaxis_title="القيمة", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("لا توجد بيانات كافية لعرض الرسم البياني")
    
    with tab_3d:
        heat_intensity = min(1, max(0, (oil_temp - 60) / 50))
        
        fig_3d = go.Figure()
        
        # هيكل المحرك
        fig_3d.add_trace(go.Mesh3d(x=[-1.6,1.6,1.6,-1.6,-1.6,1.6,1.6,-1.6],
                                   y=[-1.1,-1.1,1.1,1.1,-1.1,-1.1,1.1,1.1],
                                   z=[-1.3,-1.3,-1.3,-1.3,1.3,1.3,1.3,1.3],
                                   color='rgba(80,120,180,0.2)', opacity=0.2))
        
        # الأسطوانات
        for cx in [-0.9, -0.3, 0.3, 0.9]:
            theta = np.linspace(0, 2*np.pi, 20)
            phi = np.linspace(-0.9, 0.9, 10)
            t, p = np.meshgrid(theta, phi)
            x = cx + 0.4*np.cos(t)
            y = 0.25 + 0.4*np.sin(t)
            z = p * 1.0
            fig_3d.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0,'#bb8844'],[1,'#dd9944']], showscale=False, opacity=0.8))
        
        # البستونات
        piston_positions = [-0.85, -0.25, 0.35, 0.95]
        for i, cx in enumerate(piston_positions):
            piston_y = 0.1 + 0.3 * np.sin(time.time() * 7 + i * 1.5)
            piston_color = f'rgb({int(200 + heat_intensity*55)}, {int(100 - heat_intensity*50)}, 50)'
            fig_3d.add_trace(go.Mesh3d(x=[cx-0.18, cx+0.18, cx+0.18, cx-0.18, cx-0.18, cx+0.18, cx+0.18, cx-0.18],
                                       y=[piston_y-0.22, piston_y-0.22, piston_y+0.22, piston_y+0.22, piston_y-0.22, piston_y-0.22, piston_y+0.22, piston_y+0.22],
                                       z=[-0.6,-0.6,-0.6,-0.6,0.6,0.6,0.6,0.6],
                                       color=piston_color, opacity=0.95, name=f'بستون {i+1}'))
        
        # المروحة
        prop_speed = max(0.2, rpm / 400.0)
        prop_angle = time.time() * prop_speed * 2 * np.pi
        for blade_deg in [0, 120, 240]:
            ang = np.radians(blade_deg + (prop_angle * 180/np.pi))
            x_blade = [0, 0.5*np.cos(ang), 1.3*np.cos(ang)]
            y_blade = [0, 0.5*np.sin(ang), 1.3*np.sin(ang)]
            z_blade = [-0.18, -0.18, -0.18]
            fig_3d.add_trace(go.Scatter3d(x=x_blade, y=y_blade, z=z_blade, mode='lines+markers',
                                          line=dict(width=12, color='#ffaa55'), marker=dict(size=4, color='#ffaa55')))
        
        # التصوير الحراري
        x_therm = np.random.uniform(-1.5, 1.5, 200)
        y_therm = np.random.uniform(-0.8, 1.0, 200)
        z_therm = np.random.uniform(-1.2, 1.2, 200)
        colors = ['red' if np.random.rand() < heat_intensity else 'blue' for _ in range(200)]
        fig_3d.add_trace(go.Scatter3d(x=x_therm, y=y_therm, z=z_therm, mode='markers',
                                      marker=dict(size=3, color=colors, opacity=0.5)))
        
        fig_3d.update_layout(
            title=f"🚢 النموذج الرقمي التوأم | RPM={rpm:.0f} | حرارة={oil_temp:.1f}°C | كفاءة={efficiency:.0f}%",
            scene=dict(camera=dict(eye=dict(x=2.2, y=1.8, z=1.6)), bgcolor='#0a0a2a'),
            template='plotly_dark',
            height=700
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# تحديث تلقائي
if auto_refresh:
    time.sleep(3)
    st.rerun()

# ========================= تذييل الصفحة =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🚢 النموذج الرقمي التوأم للمحرك البحري – نظام ذكي للصيانة التنبؤية</p>
    <p>جامعة شرق بورسعيد التكنولوجية – قسم تشغيل وصيانة السفن</p>
</div>
""", unsafe_allow_html=True)