"""
🏥 Symulator Położniczy – ML w Praktyce Klinicznej
Zadanie 3 (rozszerzone) – Wersja Streamlit
Autor: na podstawie notebooka Krzysztof Gajda
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import io

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Symulator Położniczy – ML",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1a3c5e 0%, #2e75b6 100%);
    color: white;
    padding: 20px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
  }
  .main-header h1 { margin: 0 0 6px; font-size: 1.7rem; }
  .main-header p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }
  .result-box {
    padding: 16px; border-radius: 10px; margin-top: 12px;
    text-align: center; font-weight: bold;
  }
  .scenario-table { font-size: 0.85rem; }
  div[data-testid="stTabs"] button {
    font-weight: bold !important;
    font-size: 1rem !important;
  }
  .stSlider > div > div > div { background: #2e75b6 !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
col_header, col_btn = st.columns([5, 1])
with col_header:
    st.markdown("""
<div class="main-header">
  <h1>🤰 Symulator Położniczy – ML w Praktyce Klinicznej</h1>
  <p>Zadanie. Położnictwo II stopień: aplikacja tylko dla organizacji tych zajęć. Autor: Krzysztof Gajda</p>
</div>
""", unsafe_allow_html=True)
with col_btn:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Początek", use_container_width=True, help="Resetuje wszystkie suwaki do wartości domyślnych"):
        st.rerun()

# ════════════════════════════════════════════════════════════
# MODEL TRAINING (cached)
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚙️ Trening modeli Random Forest (jednorazowo)...")
def train_models():
    np.random.seed(42)
    N = 3000

    # ── KTG ─────────────────────────────────────────────────
    fhr_base    = np.random.uniform(100, 175, N)
    akceleracje = np.random.poisson(3, N).clip(0, 12)
    decel_wczesne = np.random.poisson(0.5, N).clip(0, 8)
    decel_pozne  = np.random.poisson(0.3, N).clip(0, 6)
    decel_zm     = np.random.poisson(0.4, N).clip(0, 8)
    stv          = np.random.uniform(0.5, 18, N)
    ltv          = np.random.uniform(0, 50, N)
    ruchy_plodu  = np.random.poisson(6, N).clip(0, 20)
    skurcze_mac  = np.random.poisson(2, N).clip(0, 10)
    tydzien_ktg  = np.random.uniform(28, 42, N)

    score_ktg = np.zeros(N)
    score_ktg += np.where((fhr_base < 110) | (fhr_base > 160), 1.5, 0)
    score_ktg += np.where(akceleracje == 0, 1.2, 0)
    score_ktg += np.where(akceleracje >= 2, -0.5, 0)
    score_ktg += decel_pozne * 1.8
    score_ktg += decel_zm * 0.9
    score_ktg += np.where(stv < 3, 2.0, 0)
    score_ktg += np.where(stv < 5, 0.8, 0)
    score_ktg += np.where(stv > 8, -0.3, 0)
    score_ktg += np.where(ruchy_plodu == 0, 0.8, 0)
    score_ktg += np.random.normal(0, 0.5, N)
    y_ktg = np.where(score_ktg < 1.5, 1, np.where(score_ktg < 3.5, 2, 3))

    df_ktg = pd.DataFrame({
        'fhr_baseline': fhr_base, 'akceleracje': akceleracje,
        'decel_wczesne': decel_wczesne, 'decel_pozne': decel_pozne,
        'decel_zmienne': decel_zm, 'stv_ms': stv, 'ltv_ms': ltv,
        'ruchy_plodu': ruchy_plodu, 'skurcze_macicy': skurcze_mac,
        'tydzien_ciazy': tydzien_ktg, 'klasa_figo': y_ktg
    })
    feats_ktg = ['fhr_baseline','akceleracje','decel_wczesne','decel_pozne',
                 'decel_zmienne','stv_ms','ltv_ms','ruchy_plodu','skurcze_macicy','tydzien_ciazy']
    sc_ktg = StandardScaler()
    X_ktg_sc = sc_ktg.fit_transform(df_ktg[feats_ktg])
    rf_ktg = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced')
    rf_ktg.fit(X_ktg_sc, df_ktg['klasa_figo'])

    # ── APGAR ────────────────────────────────────────────────
    tydzien_ap  = np.random.uniform(24, 42, N)
    masa_g      = np.random.normal(3200, 600, N).clip(400, 5500)
    kolor_wod   = np.random.choice([0,1,2], N, p=[0.72,0.16,0.12])
    dlugosc_por = np.random.exponential(8, N).clip(0.2, 50)
    ciec_ces    = np.random.choice([0,1], N, p=[0.55,0.45])
    nadcisnienie= np.random.choice([0,1], N, p=[0.82,0.18])
    cukrzyca_ap = np.random.choice([0,1], N, p=[0.87,0.13])
    palenie_ap  = np.random.choice([0,1], N, p=[0.80,0.20])
    parzystosc  = np.random.choice([0,1,2], N, p=[0.40,0.40,0.20])
    ph_pepowiny = np.random.normal(7.28, 0.08, N).clip(6.8, 7.5)

    score_apgar = (
        0.7*((tydzien_ap-24)/18) + 0.5*(masa_g/5000) - 0.6*(kolor_wod/2)
        - 0.4*(dlugosc_por/30) + 0.3*ciec_ces - 0.35*nadcisnienie
        - 0.25*cukrzyca_ap - 0.25*palenie_ap
        + 0.5*((ph_pepowiny-6.8)/0.7) + np.random.normal(0,0.3,N)
    )
    apgar_raw = np.clip(2+8*(1/(1+np.exp(-score_apgar*2))),0,10).round(0).astype(int)
    apgar_kat  = np.where(apgar_raw>=7, 2, np.where(apgar_raw>=4, 1, 0))

    df_apgar = pd.DataFrame({
        'tydzien_ciazy': tydzien_ap, 'masa_urodzeniowa_g': masa_g,
        'kolor_wod_plodowych': kolor_wod, 'dlugosc_porodu_h': dlugosc_por,
        'ciecie_cesarskie': ciec_ces, 'nadcisnienie_ciazy': nadcisnienie,
        'cukrzyca_ciazowa': cukrzyca_ap, 'palenie_w_ciazy': palenie_ap,
        'parzystosc': parzystosc, 'ph_pepowiny': ph_pepowiny,
        'apgar_wynik': apgar_raw, 'apgar_kategoria': apgar_kat
    })
    feats_apgar = ['tydzien_ciazy','masa_urodzeniowa_g','kolor_wod_plodowych',
                   'dlugosc_porodu_h','ciecie_cesarskie','nadcisnienie_ciazy',
                   'cukrzyca_ciazowa','palenie_w_ciazy','parzystosc','ph_pepowiny']
    sc_ap = StandardScaler()
    X_ap_sc = sc_ap.fit_transform(df_apgar[feats_apgar])
    rf_ap_kat = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced')
    rf_ap_kat.fit(X_ap_sc, df_apgar['apgar_kategoria'])
    rf_ap_reg = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    rf_ap_reg.fit(X_ap_sc, df_apgar['apgar_wynik'])

    # ── PPT ──────────────────────────────────────────────────
    wiek_m_p    = np.random.normal(28,6,N).clip(15,46).astype(int)
    dlug_szyjki = np.random.normal(36,13,N).clip(3,60)
    ffn         = np.random.choice([0,1], N, p=[0.74,0.26])
    ppp_wywiad  = np.random.choice([0,1], N, p=[0.87,0.13])
    infekcja_p  = np.random.choice([0,1], N, p=[0.77,0.23])
    mnoga_p     = np.random.choice([0,1], N, p=[0.96,0.04])
    palenie_p   = np.random.choice([0,1], N, p=[0.82,0.18])
    crp_p       = np.random.exponential(4,N).clip(0.1,60)
    bmi_p       = np.random.normal(24,4.5,N).clip(15,46)
    tydzien_wiz = np.random.uniform(14,34,N)
    stres_p     = np.random.choice([0,1,2], N, p=[0.50,0.35,0.15])

    score_ppt = (
        1.3*ppp_wywiad + 1.1*mnoga_p + 1.2*(dlug_szyjki<25) + 0.8*(dlug_szyjki<15)
        + 0.9*ffn + 0.6*infekcja_p + 0.35*palenie_p
        + 0.4*(wiek_m_p<18) + 0.3*(wiek_m_p>40)
        + 0.3*(stres_p==2) + 0.3*np.log1p(crp_p)/4
        + np.random.normal(0,0.55,N)
    )
    prob_ppt = 1/(1+np.exp(-(score_ppt-1.4)))
    y_ppt = (np.random.random(N) < prob_ppt).astype(int)

    df_ppt = pd.DataFrame({
        'wiek_matki': wiek_m_p, 'dlug_szyjki_mm': dlug_szyjki,
        'fibronektyna_fFN': ffn, 'ppp_wywiad': ppp_wywiad,
        'infekcja_pochwy': infekcja_p, 'ciaza_mnoga': mnoga_p,
        'palenie': palenie_p, 'crp_mgL': crp_p, 'bmi': bmi_p,
        'tydzien_wizyty': tydzien_wiz, 'stres': stres_p, 'ppt': y_ppt
    })
    feats_ppt = ['wiek_matki','dlug_szyjki_mm','fibronektyna_fFN','ppp_wywiad',
                 'infekcja_pochwy','ciaza_mnoga','palenie','crp_mgL','bmi','tydzien_wizyty','stres']
    sc_ppt = StandardScaler()
    X_ppt_sc = sc_ppt.fit_transform(df_ppt[feats_ppt])
    rf_ppt = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced')
    rf_ppt.fit(X_ppt_sc, df_ppt['ppt'])

    return (sc_ktg, rf_ktg, feats_ktg,
            sc_ap, rf_ap_kat, rf_ap_reg, feats_apgar,
            sc_ppt, rf_ppt, feats_ppt)


# Load models
(sc_ktg, rf_ktg, feats_ktg,
 sc_ap, rf_ap_kat, rf_ap_reg, feats_apgar,
 sc_ppt, rf_ppt, feats_ppt) = train_models()


# ════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ════════════════════════════════════════════════════════════

def gen_ktg_fig(fhr, akc, decW, decP, decZ, stv, ltv, ruch, skur, tktg):
    fhr=float(fhr); akc=int(akc); decW=int(decW); decP=int(decP); decZ=int(decZ)
    stv=float(stv); ltv=float(ltv); ruch=int(ruch); skur=int(skur); tktg=float(tktg)

    X_in = sc_ktg.transform([[fhr,akc,decW,decP,decZ,stv,ltv,ruch,skur,tktg]])
    proba = rf_ktg.predict_proba(X_in)[0]
    klasa = rf_ktg.predict(X_in)[0]
    klasa_labels = {1:('PRAWIDŁOWY','#2e7d32','✅'), 2:('WĄTPLIWY','#e65100','⚠️'), 3:('NIEPRAWIDŁOWY','#b71c1c','🆘')}
    k_label, k_color, k_icon = klasa_labels[klasa]

    t = np.linspace(0, 30, 1800)
    np.random.seed(42)
    fhr_sig = np.full(len(t), fhr)
    fhr_sig += np.random.normal(0, stv*0.3, len(t))
    fhr_sig += (ltv/8)*np.sin(2*np.pi*t/25)
    for _ in range(min(akc, 4)):
        pos=np.random.randint(200,1600); w=np.random.randint(80,180); amp=np.random.uniform(12,22)
        fhr_sig[pos:pos+w] += amp*np.exp(-((np.arange(w)-w//2)**2)/(2*(w//4)**2))
    for _ in range(min(decP, 3)):
        pos=np.random.randint(400,1400); w=np.random.randint(100,220); amp=np.random.uniform(15,35)
        fhr_sig[pos:pos+w] -= amp*np.exp(-((np.arange(w)-w//2)**2)/(2*(w//4)**2))
    for _ in range(min(decZ, 3)):
        pos=np.random.randint(200,1600); w=np.random.randint(60,140); amp=np.random.uniform(20,45)
        fhr_sig[pos:pos+w] -= amp*np.exp(-((np.arange(w)-w//2)**2)/(2*(w//5)**2))
    fhr_sig = np.clip(fhr_sig, 50, 210)
    toco_sig = np.zeros(len(t))
    for i in range(min(skur, 5)):
        pos=int(i*len(t)/max(skur,1))+np.random.randint(-50,50)
        pos=max(50,min(pos,len(t)-200)); w=np.random.randint(150,300)
        toco_sig[pos:pos+w] += 40*np.exp(-((np.arange(w)-w//2)**2)/(2*(w//4)**2))

    fig = plt.figure(figsize=(13,8)); fig.patch.set_facecolor('#fafafa')
    gs = gridspec.GridSpec(3,2,figure=fig,hspace=0.55,wspace=0.35,height_ratios=[2.5,1.5,1])
    ax1 = fig.add_subplot(gs[0,:])
    ax1.set_facecolor('#fff9f0')
    for y in range(60,220,10):
        ax1.axhline(y, color=('#c9a870' if y%30==0 else '#e8d5b0'), lw=(1.5 if y%30==0 else 0.8), alpha=0.7)
    ax1.axhspan(110,160,alpha=0.06,color='green')
    ax1.plot(t, fhr_sig, color='#c0392b', lw=1.0)
    ax1.axhline(110,color='#27ae60',lw=1.2,ls='--',alpha=0.6,label='Zakres prawidłowy (110–160)')
    ax1.axhline(160,color='#27ae60',lw=1.2,ls='--',alpha=0.6)
    ax1.set_ylim(60,210); ax1.set_xlim(0,30)
    ax1.set_ylabel('FHR [udc/min]',fontsize=11,fontweight='bold')
    ax1.set_title(f'KTG – FHR: {fhr:.0f} udc/min | STV: {stv} ms | Akceleracje: {akc} | Dec. późne: {decP}',fontsize=10)
    ax1.legend(loc='upper right',fontsize=8); ax1.set_xlabel('Czas [s]',fontsize=10)
    ax2 = fig.add_subplot(gs[1,:])
    ax2.set_facecolor('#f0f4ff')
    ax2.fill_between(t,toco_sig,alpha=0.5,color='#2980b9'); ax2.plot(t,toco_sig,color='#1a5276',lw=1.2)
    ax2.set_ylim(0,100); ax2.set_xlim(0,30)
    ax2.set_ylabel('TOCO [mmHg]',fontsize=10,fontweight='bold')
    ax2.set_title(f'Skurcze macicy – {skur}/10 min',fontsize=10); ax2.set_xlabel('Czas [s]',fontsize=10)
    ax3 = fig.add_subplot(gs[2,0]); ax3.axis('off')
    bbox=FancyBboxPatch((0.02,0.05),0.96,0.90,boxstyle="round,pad=0.03",
                        facecolor=k_color,edgecolor='white',alpha=0.9,transform=ax3.transAxes)
    ax3.add_patch(bbox)
    ax3.text(0.5,0.72,f'{k_icon}  PREDYKCJA ML',ha='center',va='center',fontsize=10,color='white',fontweight='bold',transform=ax3.transAxes)
    ax3.text(0.5,0.40,k_label,ha='center',va='center',fontsize=16,color='white',fontweight='bold',transform=ax3.transAxes)
    ax3.text(0.5,0.12,f'Pewność: {max(proba)*100:.0f}%',ha='center',va='center',fontsize=10,color='white',alpha=0.9,transform=ax3.transAxes)
    ax4 = fig.add_subplot(gs[2,1]); ax4.set_facecolor('#fafafa')
    klasy=['Prawidłowy\n(FIGO 1)','Wątpliwy\n(FIGO 2)','Nieprawidłowy\n(FIGO 3)']
    bars=ax4.bar(klasy,proba*100,color=['#2e7d32','#e65100','#b71c1c'],alpha=0.85,edgecolor='white')
    for bar,p in zip(bars,proba):
        if p>0.05: ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{p*100:.0f}%',ha='center',fontsize=10,fontweight='bold')
    ax4.set_ylim(0,110); ax4.set_ylabel('Prawdopodobieństwo [%]',fontsize=9)
    ax4.set_title('Rozkład klas',fontsize=10); ax4.axhline(50,color='gray',ls='--',lw=1,alpha=0.5)
    plt.tight_layout()
    return fig, klasa, k_label, k_color, max(proba)


def gen_apgar_fig(tyg, masa, wody, por, ciec, nadc, cuk, pal, par, ph):
    tyg=float(tyg); masa=int(masa); wody=int(wody); por=float(por)
    ciec=int(ciec); nadc=int(nadc); cuk=int(cuk); pal=int(pal); par=int(par); ph=float(ph)

    X_in = sc_ap.transform([[tyg,masa,wody,por,ciec,nadc,cuk,pal,par,ph]])
    kat = rf_ap_kat.predict(X_in)[0]
    prob = rf_ap_kat.predict_proba(X_in)[0]
    apgar_val = int(np.clip(round(rf_ap_reg.predict(X_in)[0]),0,10))
    kat_info = {
        0:('Ciężki stan (0–3)','#b71c1c','RESUSCYTACJA – natychmiastowe działanie!'),
        1:('Umiarkowany (4–6)','#e65100','Stymulacja, tlen, monitorowanie'),
        2:('Stan dobry (7–10)','#2e7d32','Standardowa opieka noworodkowa')
    }
    k_label, k_color, k_action = kat_info[kat]

    fig, axes = plt.subplots(1,3,figsize=(13,5)); fig.patch.set_facecolor('#fafafa')
    ax=axes[0]; ax.set_aspect('equal'); ax.axis('off')
    theta_s=np.pi; theta_e=0
    theta_val=theta_s+(theta_e-theta_s)*(apgar_val/10)
    theta_range=np.linspace(theta_s,theta_e,300)
    for i in range(300):
        t0=theta_range[i]; frac=i/299; r=1-frac; g=frac
        ax.plot([0.9*np.cos(t0),np.cos(t0)],[0.9*np.sin(t0),np.sin(t0)],color=(r,g,0),lw=4,alpha=0.7)
    ax.annotate('',xy=(0.75*np.cos(theta_val),0.75*np.sin(theta_val)),xytext=(0,0),
                arrowprops=dict(arrowstyle='->',color=k_color,lw=3.5,mutation_scale=20))
    ax.add_patch(plt.Circle((0,0),0.08,color=k_color,zorder=5))
    for v in [0,2,4,6,7,8,10]:
        ang=theta_s+(theta_e-theta_s)*(v/10)
        ax.text(1.15*np.cos(ang),1.15*np.sin(ang),str(v),ha='center',va='center',fontsize=10,fontweight='bold')
    ax.text(0,-0.25,f'{apgar_val}',ha='center',fontsize=38,fontweight='bold',color=k_color)
    ax.text(0,-0.48,'APGAR',ha='center',fontsize=13,color='#555')
    ax.text(0,0.65,k_label,ha='center',fontsize=10,fontweight='bold',color=k_color)
    ax.set_xlim(-1.4,1.4); ax.set_ylim(-0.7,1.3); ax.set_title('Przewidywany Apgar 1 min',fontsize=11,pad=8)

    ax2=axes[1]; ax2.set_facecolor('#fafafa')
    bars=ax2.bar(['Ciężki\n(0–3)','Umiarkowany\n(4–6)','Dobry\n(7–10)'],prob*100,
                  color=['#b71c1c','#e65100','#2e7d32'],alpha=0.85,edgecolor='white',width=0.55)
    for bar,p in zip(bars,prob):
        if p>0.04: ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{p*100:.0f}%',ha='center',fontsize=11,fontweight='bold')
    ax2.set_ylim(0,112); ax2.set_ylabel('Prawdopodobieństwo [%]',fontsize=10)
    ax2.set_title('Rozkład predykcji klas Apgar',fontsize=11); ax2.axhline(50,color='gray',ls='--',lw=1,alpha=0.5)

    ax3=axes[2]; ax3.axis('off')
    ryzyko=[]
    if tyg<32: ryzyko.append(('🔴 Wcześniactwo < 32 tc','#c62828'))
    elif tyg<37: ryzyko.append(('🟠 Wcześniactwo 32–37 tc','#e65100'))
    if masa<1500: ryzyko.append(('🔴 Bardzo niska masa < 1500g','#c62828'))
    elif masa<2500: ryzyko.append(('🟠 Niska masa 1500–2500g','#e65100'))
    if wody==2: ryzyko.append(('🔴 Wody smółkowe','#c62828'))
    elif wody==1: ryzyko.append(('🟠 Wody zabarwione','#e65100'))
    if ph<7.1: ryzyko.append(('🔴 Ciężka kwasica pH < 7.10','#c62828'))
    elif ph<7.2: ryzyko.append(('🟠 Kwasica pH 7.10–7.20','#e65100'))
    if nadc: ryzyko.append(('🟡 Nadciśnienie matki','#f9a825'))
    if cuk: ryzyko.append(('🟡 Cukrzyca ciążowa','#f9a825'))
    if pal: ryzyko.append(('🟡 Palenie w ciąży','#f9a825'))
    if not ryzyko: ryzyko.append(('✅ Brak istotnych czynników ryzyka','#2e7d32'))
    ax3.text(0.05,0.95,'⚕ Czynniki ryzyka',transform=ax3.transAxes,fontsize=12,fontweight='bold',va='top',color='#1a3c5e')
    for i,(tekst,kol) in enumerate(ryzyko[:7]):
        ax3.text(0.05,0.80-i*0.11,tekst,transform=ax3.transAxes,fontsize=10,va='top',color=kol)
    ax3.text(0.05,0.05,f'➤ {k_action}',transform=ax3.transAxes,fontsize=9.5,color=k_color,fontweight='bold',va='bottom',
             bbox=dict(boxstyle='round',facecolor=k_color+'22',edgecolor=k_color))
    plt.suptitle(f'Symulator Apgar | {tyg} t.c. | {masa} g',fontsize=12,fontweight='bold',y=1.01)
    plt.tight_layout()
    return fig, apgar_val, k_label, k_action, k_color


def gen_ppt_fig(wiek, szyjka, tyg_w, crp, bmi, ffn, ppp, inf, mnog, pal, stres):
    wiek=int(wiek); szyjka=float(szyjka); tyg_w=float(tyg_w); crp=float(crp); bmi=float(bmi)
    ffn=int(ffn); ppp=int(ppp); inf=int(inf); mnog=int(mnog); pal=int(pal); stres=int(stres)

    X_in = sc_ppt.transform([[wiek,szyjka,ffn,ppp,inf,mnog,pal,crp,bmi,tyg_w,stres]])
    prob_val = rf_ppt.predict_proba(X_in)[0][1]
    if prob_val<0.20: ryzyko_str,ryzyko_kol,rek='NISKIE','#2e7d32','Standardowa kontrola. Ponowna ocena za 4 tygodnie.'
    elif prob_val<0.45: ryzyko_str,ryzyko_kol,rek='UMIARKOWANE','#e65100','Rozważ profilaktykę (progesteron). Kontrola USG za 2 tyg.'
    elif prob_val<0.70: ryzyko_str,ryzyko_kol,rek='WYSOKIE','#c62828','Hospitalizacja! Progesteron, kortykoidy dla płuca płodu.'
    else: ryzyko_str,ryzyko_kol,rek='BARDZO WYSOKIE','#7b0000','PILNA hospitalizacja! Kortykosteroidy STAT. Gotowość neonatologiczna.'

    fig = plt.figure(figsize=(14,5.5)); fig.patch.set_facecolor('#fafafa')
    ax = fig.add_subplot(131)
    ax.set_facecolor('white'); ax.set_xlim(0,1); ax.set_ylim(-0.05,1.05)
    segs=[(0.0,0.20,'#e8f5e9'),(0.20,0.45,'#fff3e0'),(0.45,0.70,'#ffebee'),(0.70,1.0,'#ffcdd2')]
    for x0,x1,c in segs:
        ax.add_patch(mpatches.FancyBboxPatch((x0,0.35),x1-x0,0.15,boxstyle="square,pad=0",facecolor=c,edgecolor='#ccc',lw=0.5))
    ax.plot([prob_val,prob_val],[0.28,0.55],color=ryzyko_kol,lw=4,solid_capstyle='round',zorder=5)
    ax.plot(prob_val,0.27,'v',color=ryzyko_kol,ms=14,zorder=6)
    for v,label in [(0.10,'Niskie'),(0.32,'Umiarkowane'),(0.57,'Wysokie'),(0.85,'B. Wysokie')]:
        ax.text(v,0.58,label,ha='center',fontsize=8,color='#555')
    for v in [0.0,0.2,0.45,0.70,1.0]:
        ax.text(v,0.30,f'{int(v*100)}%',ha='center',fontsize=8.5,color='#333')
    ax.text(0.5,0.82,f'{prob_val*100:.1f}%',ha='center',fontsize=42,fontweight='bold',color=ryzyko_kol)
    ax.text(0.5,0.92,'Ryzyko PPT',ha='center',fontsize=10,color='#333')
    ax.text(0.5,0.12,f'RYZYKO: {ryzyko_str}',ha='center',fontsize=12,fontweight='bold',color=ryzyko_kol)
    ax.axis('off'); ax.set_title('Predykcja PPT (Model ML)',fontsize=11,pad=6)

    czynniki=['Szyjka\n(<25mm)','fFN+','Poprz.\nPPT','Infekcja','Ciąża\nmnoga','CRP\n>10']
    wartosci=[max(0,min(1,(25-szyjka)/25)),float(ffn),float(ppp),float(inf),float(mnog),min(1,crp/20)]
    angles=np.linspace(0,2*np.pi,len(czynniki),endpoint=False)
    wp=wartosci+wartosci[:1]; ap=np.append(angles,angles[0])
    ax2=fig.add_subplot(132,projection='polar'); ax2.set_facecolor('#fafafa')
    ax2.plot(ap,wp,'o-',lw=2,color=ryzyko_kol); ax2.fill(ap,wp,alpha=0.25,color=ryzyko_kol)
    ax2.set_xticks(angles); ax2.set_xticklabels(czynniki,size=9)
    ax2.set_ylim(0,1); ax2.set_yticks([0.25,0.5,0.75,1.0]); ax2.set_yticklabels(['25%','50%','75%','100%'],size=7)
    ax2.set_title('Profil czynników ryzyka',fontsize=11,pad=14)

    ax3=fig.add_subplot(133); ax3.axis('off')
    ax3.add_patch(FancyBboxPatch((0.03,0.03),0.94,0.94,boxstyle="round,pad=0.03",
                  facecolor=ryzyko_kol+'18',edgecolor=ryzyko_kol,lw=2,transform=ax3.transAxes))
    ax3.text(0.5,0.91,'⚕ Rekomendacja kliniczna',ha='center',va='top',fontsize=11,fontweight='bold',color=ryzyko_kol,transform=ax3.transAxes)
    ax3.text(0.5,0.75,rek,ha='center',va='top',fontsize=10,color='#222',transform=ax3.transAxes,multialignment='center')
    szczeg=[]
    if szyjka<25: szczeg.append(f'• Szyjka {szyjka:.0f} mm → KRÓTKA (próg 25 mm)')
    if ffn: szczeg.append('• fFN POZYTYWNA → wysokie ryzyko 2-tyg.')
    if ppp: szczeg.append('• Poprzedni PPT → 15-30% ryzyko nawrotu')
    if inf: szczeg.append('• Infekcja → leczenie antybiotykami')
    if mnog: szczeg.append('• Ciąża mnoga → wyższe ryzyko bazowe')
    if crp>10: szczeg.append(f'• CRP {crp:.0f} mg/L → stan zapalny')
    if not szczeg: szczeg.append('• Brak istotnych czynników wysokiego ryzyka')
    for i,s in enumerate(szczeg[:5]):
        ax3.text(0.08,0.55-i*0.10,s,va='top',fontsize=9.5,color='#333',transform=ax3.transAxes)
    ax3.set_title('Interpretacja i postępowanie',fontsize=11,pad=6)
    plt.suptitle(f'Symulator PPT | {tyg_w:.1f} t.c. | Szyjka: {szyjka:.0f} mm',fontsize=12,fontweight='bold',y=1.01)
    plt.tight_layout()
    return fig, prob_val, ryzyko_str, rek, ryzyko_kol


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 KTG", "👶 Apgar", "🤰 Poród przedwczesny", "📋 Scenariusze & Pytania", "🤖 O modelach"])


# ────────────────────────────────────────────────────────────
# TAB 1: KTG
# ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Symulator KTG – Kardiotokografia")
    st.markdown("Ustaw parametry zapisu KTG i wciśnij **▶ Generuj KTG** aby zobaczyć predykcję modelu AI (klasyfikacja FIGO 2015).")

    col1, col2 = st.columns(2)
    with col1:
        fhr  = st.slider("FHR baseline [udc/min]", 90, 180, 135, 1)
        akc  = st.slider("Akceleracje [/10 min]", 0, 12, 3, 1)
        decW = st.slider("Deceleracje wczesne", 0, 8, 0, 1)
        decP = st.slider("Deceleracje późne ⚠️", 0, 8, 0, 1)
        decZ = st.slider("Deceleracje zmienne", 0, 8, 0, 1)
    with col2:
        stv  = st.slider("STV – zmienność krótkoterminowa [ms]", 0.5, 18.0, 7.0, 0.5)
        ltv  = st.slider("LTV – zmienność długoterminowa [ms]", 0, 50, 15, 1)
        ruch = st.slider("Ruchy płodu [/30 min]", 0, 20, 5, 1)
        skur = st.slider("Skurcze macicy [/10 min]", 0, 10, 2, 1)
        tktg = st.slider("Tydzień ciąży [tc]", 26.0, 42.0, 38.0, 0.5)

    if st.button("▶ Generuj KTG", type="primary", use_container_width=True):
        with st.spinner("Generowanie wykresu KTG..."):
            fig, klasa, k_label, k_color, pewnosc = gen_ktg_fig(fhr,akc,decW,decP,decZ,stv,ltv,ruch,skur,tktg)
            color_map = {'#2e7d32':'🟢', '#e65100':'🟠', '#b71c1c':'🔴'}
            icon = color_map.get(k_color, '⚪')
            st.markdown(f"""
            <div style='background:{k_color};color:white;padding:14px 20px;border-radius:10px;text-align:center;margin-bottom:12px'>
              <span style='font-size:1.3rem;font-weight:bold'>{icon} FIGO Klasa {klasa}: {k_label}</span>
              <br><span style='opacity:0.9'>Pewność modelu: {pewnosc*100:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(fig)
            plt.close(fig)


# ────────────────────────────────────────────────────────────
# TAB 2: APGAR
# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("👶 Symulator Apgar – Ocena stanu noworodka")
    st.markdown("Ustaw parametry kliniczne porodu i wciśnij **▶ Oblicz Apgar** aby zobaczyć predykcję modelu AI.")

    col1, col2 = st.columns(2)
    with col1:
        tyg_ap  = st.slider("Tydzień ciąży [tc]", 24.0, 42.0, 39.0, 0.5, key="ap_tyg")
        masa    = st.slider("Masa urodzeniowa [g]", 400, 5500, 3300, 50)
        por_h   = st.slider("Długość porodu [h]", 0.2, 50.0, 8.0, 0.2)
        ph      = st.slider("pH krwi pępowiny", 6.80, 7.50, 7.28, 0.01)
    with col2:
        wody    = st.selectbox("Wody płodowe", options=[0,1,2],
                                format_func=lambda x: ["Czyste","Zabarwione krwią","Smółkowe"][x])
        par     = st.selectbox("Parzystość", options=[0,1,2],
                                format_func=lambda x: ["Pierworódka","II poród","≥ III poród"][x])
        ciec    = st.checkbox("Cięcie cesarskie")
        nadc    = st.checkbox("Nadciśnienie w ciąży")
        cuk     = st.checkbox("Cukrzyca ciążowa")
        pal_a   = st.checkbox("Palenie w ciąży")

    if st.button("▶ Oblicz Apgar", type="primary", use_container_width=True):
        with st.spinner("Obliczanie Apgar..."):
            fig, apgar_val, k_label, k_action, k_color = gen_apgar_fig(
                tyg_ap, masa, wody, por_h, int(ciec), int(nadc), int(cuk), int(pal_a), par, ph)
            st.markdown(f"""
            <div style='background:{k_color};color:white;padding:14px 20px;border-radius:10px;text-align:center;margin-bottom:12px'>
              <span style='font-size:1.3rem;font-weight:bold'>Apgar = {apgar_val} · {k_label}</span>
              <br><span style='opacity:0.9'>➤ {k_action}</span>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(fig)
            plt.close(fig)


# ────────────────────────────────────────────────────────────
# TAB 3: PPT
# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🤰 Symulator PPT – Ryzyko porodu przedwczesnego")
    st.markdown("Ustaw parametry kliniczne i wciśnij **▶ Oblicz ryzyko PPT** aby zobaczyć predykcję modelu AI.")

    col1, col2 = st.columns(2)
    with col1:
        wiek_m   = st.slider("Wiek matki [lata]", 15, 46, 28, 1)
        szyjka   = st.slider("Długość szyjki macicy [mm]", 3.0, 60.0, 38.0, 0.5)
        tyg_w    = st.slider("Tydzień ciąży (wizyta) [tc]", 14.0, 34.0, 24.0, 0.5)
        crp_val  = st.slider("CRP [mg/L]", 0.1, 60.0, 2.0, 0.5)
        bmi_val  = st.slider("BMI przed ciążą", 15.0, 46.0, 24.0, 0.5)
    with col2:
        stres_p  = st.selectbox("Poziom stresu", options=[0,1,2],
                                  format_func=lambda x: ["Brak","Umiarkowany","Duży"][x])
        ffn_p    = st.checkbox("Fibronektyna płodowa (fFN) POZYTYWNA")
        ppp_p    = st.checkbox("Poprzedni poród przedwczesny")
        inf_p    = st.checkbox("Infekcja pochwy / szyjki")
        mnog_p   = st.checkbox("Ciąża mnoga")
        pal_p    = st.checkbox("Palenie papierosów")

    if st.button("▶ Oblicz ryzyko PPT", type="primary", use_container_width=True):
        with st.spinner("Obliczanie ryzyka PPT..."):
            fig, prob_val, ryzyko_str, rek, ryzyko_kol = gen_ppt_fig(
                wiek_m, szyjka, tyg_w, crp_val, bmi_val,
                int(ffn_p), int(ppp_p), int(inf_p), int(mnog_p), int(pal_p), stres_p)
            st.markdown(f"""
            <div style='background:{ryzyko_kol};color:white;padding:14px 20px;border-radius:10px;text-align:center;margin-bottom:12px'>
              <span style='font-size:1.3rem;font-weight:bold'>Ryzyko PPT: {prob_val*100:.1f}% – {ryzyko_str}</span>
              <br><span style='opacity:0.9'>➤ {rek}</span>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(fig)
            plt.close(fig)


# ────────────────────────────────────────────────────────────
# TAB 4: SCENARIUSZE & PYTANIA
# ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Scenariusze kliniczne do odtworzenia")
    st.markdown("""
Ustaw parametry w symulatorze zgodnie z poniższymi scenariuszami i zapisz wynik modelu w tabeli obserwacji.
""")

    st.markdown("""
| Nr | Scenariusz | Parametry do ustawienia | Twoja notacja |
|----|-----------|------------------------|--------------|
| **K-1** | Prawidłowy zapis KTG | FHR 135, STV 8ms, AKC 4, brak deceleracji | Klasa FIGO: ____ |
| **K-2** | Bradykardia płodowa | FHR 95, STV 4ms, AKC 0 | Klasa FIGO: ____ |
| **K-3** | Deceleracje późne + niska STV | FHR 155, DEC_PÓŹNE 3, STV 2.5ms | Klasa FIGO: ____ |
| **K-4** | Ciężkie zagrożenie płodu | FHR 170, STV 1.5ms, AKC 0, DEC_PÓŹNE 5 | Klasa FIGO: ____ |
| **A-1** | Donoszone, bez czynników ryzyka | 39 tc, 3400g, czyste wody, pH 7.32 | Apgar: ____ |
| **A-2** | Wcześniak ze smółką | 32 tc, 1700g, wody smółkowe, pH 7.15 | Apgar: ____ |
| **P-1** | Niskie ryzyko PPT | Szyjka 42mm, fFN–, bez infekcji, 24 tc | Ryzyko PPT: ___% |
| **P-2** | Wysokie ryzyko PPT | Szyjka 18mm, fFN+, poprzedni PPT, 26 tc | Ryzyko PPT: ___% |
""")

    st.divider()
    st.subheader("❓ Pytania obowiązkowe do sprawozdania")

    with st.expander("1. KTG – wpływ parametrów na klasyfikację FIGO"):
        st.markdown("""
**Pytanie:** Który parametr miał największy wpływ na zmianę klasyfikacji z FIGO 1 na FIGO 3? Jaki ma to sens kliniczny?

*Wskazówka: Porównaj scenariusze K-1 i K-4. Zmieniaj po jednym parametrze na raz.*
""")

    with st.expander("2. KTG – izolowana niska STV"):
        st.markdown("""
**Pytanie:** Ustaw FHR = 120, STV = 2 ms, AKC = 0, DEC_PÓŹNE = 0. Czy wynik to FIGO 2 czy 3? Co oznacza izolowana niska STV bez deceleracji późnych?

*Wskazówka: Niska STV (<3 ms) jest markerem kwasicy płodowej wg FIGO 2015.*
""")

    with st.expander("3. APGAR – wpływ wód smółkowych i pH"):
        st.markdown("""
**Pytanie:** Porównaj dwa noworodki:
- (a) 38 tc, 3000g, wody czyste, pH 7.30
- (b) 38 tc, 3000g, wody smółkowe, pH 7.15

Jak bardzo zmienił się Apgar? Który czynnik miał większy wpływ?
""")

    with st.expander("4. PPT – próg długości szyjki macicy"):
        st.markdown("""
**Pytanie:** Znajdź minimalną długość szyjki macicy (przy fFN– i bez innych czynników ryzyka), przy której model przekracza próg 45% ryzyka PPT. Ile wynosi? Porównaj z wytycznymi klinicznymi (próg 25 mm).
""")

    with st.expander("5. PPT – wartość predykcyjna fFN"):
        st.markdown("""
**Pytanie:** Ustaw szyjkę 30 mm i fFN–. Następnie zmień na fFN+. O ile wzrosło ryzyko? Co to mówi o wartości predykcyjnej fibronektyny płodowej?
""")

    with st.expander("⭐ Zadanie zaawansowane"):
        st.markdown("""
Opracuj własny scenariusz kliniczny (opis przypadku ~5 zdań). Ustaw odpowiednie parametry w symulatorze, wykonaj zrzut ekranu wykresu i skomentuj wynik modelu z perspektywy klinicysty.

**Pytania do refleksji:**
- Czy model podjął właściwą decyzję?
- Co mógłby przeoczyć?
- Jakie są ograniczenia modeli ML w położnictwie?
""")

    st.divider()
    st.info("""
**📌 Tabela obserwacji** – przepisz wyniki do własnego sprawozdania:

| Nr | Scenariusz | Predykcja modelu | Ocena kliniczna |
|----|-----------|-----------------|----------------|
| K-1 | Prawidłowy KTG | | |
| K-2 | Bradykardia płodowa | | |
| K-3 | Deceleracje późne + niska STV | | |
| K-4 | Ciężkie zagrożenie płodu | | |
| A-1 | Donoszone, bez ryzyka | | |
| A-2 | Wcześniak ze smółką | | |
| P-1 | Niskie ryzyko PPT | | |
| P-2 | Wysokie ryzyko PPT | | |
""")

# ────────────────────────────────────────────────────────────
# TAB 5: O MODELACH
# ────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🤖 Jak działają modele AI w tym symulatorze?")
    st.markdown("""
> Poniższy opis jest skierowany do studentów medycyny i położnictwa —
> bez konieczności znajomości programowania czy matematyki.
""")
    st.divider()

    with st.expander("🧠 Czym jest uczenie maszynowe (Machine Learning)?", expanded=True):
        st.markdown("""
**Uczenie maszynowe** to sposób, w jaki komputer uczy się rozpoznawać wzorce
— podobnie jak lekarz, który po wielu latach praktyki potrafi szybko ocenić
zapis KTG „na pierwszy rzut oka".

Zamiast programować reguły ręcznie (np. *„jeśli FHR < 110 → podejrzane"*),
pokazujemy modelowi **tysiące przykładów** z odpowiedziami i pozwalamy mu
samodzielnie odkryć, co jest ważne.

W tym symulatorze każdy z trzech modeli (KTG, Apgar, PPT) „widział" podczas
nauki **3000 wirtualnych przypadków klinicznych** wraz z poprawnymi klasyfikacjami.
""")

    with st.expander("🌳 Co to jest Random Forest (Las Losowy)?"):
        st.markdown("""
**Random Forest** to metoda, która buduje wiele **drzew decyzyjnych** i łączy ich wyniki — stąd nazwa „las".

Wyobraź sobie konsylium lekarskie:
- Każdy lekarz (= jedno drzewo) analizuje przypadek i wydaje opinię
- Na końcu liczymy głosy — wygrywa diagnoza, którą wskazała **większość**
- Im więcej lekarzy w konsylium, tym bardziej wiarygodny wynik

W naszym symulatorze każdy model składa się z **150 drzew decyzyjnych**.
Pasek „Pewność modelu: X%" pokazuje, ile procent drzew zgodziło się z podaną klasyfikacją.

| Pewność | Interpretacja |
|---------|--------------|
| > 80% | Model jest bardzo pewny |
| 60–80% | Umiarkowana pewność |
| < 60% | Przypadek graniczny, wymaga oceny klinicznej |
""")

    with st.expander("📊 Skąd pochodzą dane treningowe?"):
        st.markdown("""
Dane użyte do treningu są **syntetyczne** — wygenerowane komputerowo według reguł klinicznych, a nie zebrane od prawdziwych pacjentek.

**Dlaczego syntetyczne?**
- Dane medyczne są objęte ścisłą ochroną prywatności (RODO)
- Pozwala to dokładnie kontrolować rozkład przypadków
- Umożliwia tworzenie trudnych scenariuszy klinicznych (np. ciężka kwasica)

**Reguły generowania** są oparte na wytycznych klinicznych:

| Model | Liczba przypadków treningowych | Źródło reguł |
|-------|-------------------------------|-------------|
| KTG   | 3 000 | FIGO 2015 |
| Apgar | 3 000 | PTG / WHO |
| PPT   | 3 000 | Wytyczne kliniczne |
""")

    with st.expander("⚕️ Co przewiduje każdy z trzech modeli?"):
        st.markdown("""
### 📊 Model KTG
Klasyfikuje zapis KTG do jednej z trzech klas wg **FIGO 2015**:
- **Klasa I (Prawidłowy)** — kontynuuj monitorowanie
- **Klasa II (Wątpliwy)** — zwiększ czujność, rozważ interwencję
- **Klasa III (Nieprawidłowy)** — natychmiastowe działanie kliniczne

Najważniejsze parametry wejściowe: FHR baseline, STV, deceleracje późne.

---
### 👶 Model Apgar
Przewiduje **wynik w skali Apgar w 1. minucie życia** (0–10) oraz kategorię stanu noworodka:
- **7–10** → stan dobry
- **4–6** → umiarkowany, wymaga stymulacji
- **0–3** → ciężki, konieczna resuscytacja

Najważniejsze parametry: tydzień ciąży, masa urodzeniowa, pH pępowiny, wody płodowe.

---
### 🤰 Model PPT
Szacuje **prawdopodobieństwo porodu przedwczesnego** (przed 37. tygodniem):
- **< 20%** → niskie ryzyko
- **20–45%** → umiarkowane
- **45–70%** → wysokie
- **> 70%** → bardzo wysokie

Najważniejsze parametry: długość szyjki macicy, fibronektyna płodowa (fFN), wywiad PPT.
""")

    with st.expander("⚠️ Ograniczenia modelu — czego AI nie wie?"):
        st.markdown("""
Modele w tym symulatorze są **narzędziem edukacyjnym**, a nie klinicznym systemem wspomagania decyzji.

**Czego model nie uwzględnia:**
- Dynamiki zmian w czasie (jeden wynik KTG to „zdjęcie", nie ciągły zapis)
- Kontekstu klinicznego (wcześniejsze badania, wywiad rodzinny)
- Rzadkich patologii nieobecnych w danych treningowych
- Złożonych interakcji między wieloma czynnikami ryzyka jednocześnie

**Złota zasada:** Model może sugerować, ale **decyzję kliniczną zawsze podejmuje lekarz**
na podstawie pełnego obrazu pacjentki.

---
> 💡 **Pytanie do refleksji:** Czy potrafisz skonstruować scenariusz kliniczny,
> w którym model myli się, a Ty jako klinicysta podjąłbyś inną decyzję?
""")

# ── FOOTER ───────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.82rem'>
  🏥 Symulator Położniczy opracowany na potrzeby zajęć: SI i ML w medycynie<br>
  ⚠️ <b>Tylko do celów edukacyjnych.</b> Nie zastępuje oceny klinicznej ani wytycznych FIGO/PTG. autor: Krzysztof Gajda.
</div>
""", unsafe_allow_html=True)
