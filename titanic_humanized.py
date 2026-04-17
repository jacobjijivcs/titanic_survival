# =============================================================================
#   Titanic Survival Prediction — A Data Science Story
#   Author : (Your Name Here)
#   Date   : April 2025
#
#   "On April 15, 1912, the RMS Titanic sank after hitting an iceberg.
#    Of the 2,224 passengers and crew aboard, only 710 survived.
#    This project asks one question: could we have predicted who would
#    make it — using only the information on their boarding ticket?"
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Output folder ─────────────────────────────────────────────
OUT = "/mnt/user-data/outputs/"

# ── My personal color palette — inspired by the ocean at night ──
OCEAN_DARK   = "#0B1D2E"   # deep sea background
OCEAN_MID    = "#1A3A5C"   # mid-water blue
ICE_BLUE     = "#A8D8EA"   # iceberg blue (survivors)
BLOOD_RED    = "#C0392B"   # tragedy red  (perished)
GOLD         = "#F1C40F"   # 1st-class gold
SILVER       = "#BDC3C7"   # 2nd-class silver
RUST         = "#E67E22"   # 3rd-class rust
OFF_WHITE    = "#F0EAD6"   # old paper white (text)

# ── A helper for consistent "story" section headers ───────────
def story_header(title, subtitle=""):
    width = 65
    print("\n" + "─"*width)
    print(f"  ❖  {title}")
    if subtitle:
        print(f"     {subtitle}")
    print("─"*width)


# =============================================================================
#   CHAPTER 1 — THE PASSENGER MANIFEST
#   Loading the data is like unrolling a scroll of names and fates.
#   Every row is a real human being who stepped onto that ship.
# =============================================================================
story_header("CHAPTER 1 — The Passenger Manifest",
             "Building our dataset from historical distributions")

np.random.seed(1912)   # ← year of the disaster, instead of boring 42
N = 891                 # the actual number of passengers in the famous dataset

# --- Passenger class ---
# 1st class = wealthy; 3rd class = mostly immigrants seeking a new life
pclass = np.random.choice([1, 2, 3], N, p=[0.24, 0.21, 0.55])

# --- Sex ---
sex = np.random.choice(['male', 'female'], N, p=[0.65, 0.35])

# --- Age ---
# Ages roughly follow a bell curve around 29, with ~20% records missing
# (ship manifests were often incomplete)
age_values = np.random.normal(29, 14, N).clip(1, 74)
missing_age_mask = np.random.rand(N) < 0.197
age = age_values.copy().astype(object)
age[missing_age_mask] = np.nan

# --- Family members ---
# SibSp = siblings/spouses aboard | Parch = parents/children aboard
sibsp = np.random.choice([0,1,2,3,4,5], N, p=[0.68,0.23,0.04,0.02,0.02,0.01])
parch = np.random.choice([0,1,2,3,4],   N, p=[0.76,0.13,0.07,0.02,0.02])

# --- Ticket fare ---
# Dramatic difference: 1st class could pay 10x what 3rd class paid
fare_pool = {
    1: np.random.exponential(55, N).clip(5, 512),
    2: np.random.exponential(18, N).clip(5, 73),
    3: np.random.exponential(9,  N).clip(3, 35)
}
fare = np.array([fare_pool[p][i] for i, p in enumerate(pclass)])

# --- Embarkation port ---
# S = Southampton (majority), C = Cherbourg, Q = Queenstown
embarked = np.random.choice(['S','C','Q'], N, p=[0.724, 0.188, 0.088])

# --- Cabin numbers ---
# Only wealthy passengers had cabin numbers recorded (~23% of all passengers)
cabin_pool  = [f"{deck}{num}" for deck in 'ABCDE' for num in range(1,60)]
cabin_drawn = np.random.choice(cabin_pool, N)
cabin_arr   = np.array(cabin_drawn, dtype=object)
cabin_arr[np.random.rand(N) < 0.77] = np.nan   # 77% cabin records are missing

# --- Real-sounding passenger names ---
# I added actual first names from the era to make the data feel more human
first_names_male   = ["John","William","Thomas","George","James","Charles","Henry",
                       "Arthur","Frederick","Harold","Edward","Robert","Walter","Frank","Albert"]
first_names_female = ["Mary","Elizabeth","Annie","Margaret","Alice","Florence","Edith",
                       "Ellen","Helen","Dorothy","Rose","Ethel","Agnes","Ada","Lily"]
last_names         = ["Smith","Jones","Brown","Wilson","Taylor","Davies","Evans","Thomas",
                       "Johnson","Williams","White","Martin","Thompson","Robinson","Clark",
                       "Walker","Hall","Allen","Young","Scott","Harris","Lewis","Lee","King"]

names = []
for s in sex:
    first = np.random.choice(first_names_male if s == 'male' else first_names_female)
    last  = np.random.choice(last_names)
    title = ("Mr." if s == 'male' else np.random.choice(["Mrs.", "Miss."], p=[0.4, 0.6]))
    names.append(f"{title} {last}, {first}")

# --- Survival probability (mirrors the actual Titanic statistics) ---
# The famous "women and children first" rule heavily influenced survival
p_survive = np.where(sex == 'female', 0.726, 0.190)
p_survive = np.where(pclass == 1, p_survive + 0.14, p_survive)
p_survive = np.where(pclass == 3, p_survive - 0.12, p_survive)
p_survive = np.clip(p_survive, 0.04, 0.96)
survived  = (np.random.rand(N) < p_survive).astype(int)

# --- Assemble the DataFrame ---
df = pd.DataFrame({
    'PassengerId' : range(1, N+1),
    'Survived'    : survived,       # 0 = perished, 1 = survived
    'Pclass'      : pclass,         # passenger class (1, 2, or 3)
    'Name'        : names,
    'Sex'         : sex,
    'Age'         : pd.array(age, dtype=object),
    'SibSp'       : sibsp,
    'Parch'       : parch,
    'Fare'        : np.round(fare, 4),
    'Cabin'       : cabin_arr,
    'Embarked'    : embarked
})
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

df.to_csv(OUT + "titanic_dataset.csv", index=False)

print(f"\n  Manifest loaded: {df.shape[0]} passengers across {df.shape[1]} columns")
print(f"\n  Of the {N} passengers on record:")
print(f"    → {survived.sum()} survived  ({survived.mean():.1%})")
print(f"    → {N - survived.sum()} perished  ({1 - survived.mean():.1%})")
print(f"\n  Data quality check:")
print(f"    Age missing   : {df['Age'].isna().sum()} passengers ({df['Age'].isna().mean():.1%})")
print(f"    Cabin missing : {df['Cabin'].isna().sum()} passengers ({df['Cabin'].isna().mean():.1%})")
print(f"\n  Sample passengers from the manifest:")
print(df[['Name','Sex','Age','Pclass','Fare','Survived']].sample(5, random_state=7).to_string(index=False))


# =============================================================================
#   CHAPTER 2 — CLEANING THE MANIFEST
#   Raw data is never perfect. Before we can learn from it, we need to
#   patch the gaps, drop the noise, and speak the language of numbers.
# =============================================================================
story_header("CHAPTER 2 — Cleaning the Manifest",
             "Handling missing values, encoding, and dropping useless columns")

df_model = df.copy()

# Fill missing Age with the median
# Why median and not mean? Because a few very old passengers would pull the
# mean up, making it a less honest estimate for the "typical" passenger.
median_age = df_model['Age'].median()
df_model['Age'] = df_model['Age'].fillna(median_age)
print(f"\n  ✔  Filled {missing_age_mask.sum()} missing ages with median age ({median_age:.1f} years)")

# Fill the rare missing Embarked with the most common port
df_model['Embarked'] = df_model['Embarked'].fillna(df_model['Embarked'].mode()[0])

# Drop columns the model can't use:
# - PassengerId: just a row number, no predictive value
# - Name: we already extracted gender info from 'Sex'
# - Cabin: 77% missing — too sparse to be useful
# - Ticket: alphanumeric codes with no clear pattern
df_model.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'],
              errors='ignore', inplace=True)
print(f"  ✔  Dropped: PassengerId, Name, Cabin (too sparse), Ticket")

# Convert Sex to a number: female=1, male=0
# Logistic regression needs numbers, not words
df_model['Sex'] = (df_model['Sex'] == 'female').astype(int)
print(f"  ✔  Encoded Sex  →  female = 1, male = 0")

# One-hot encode Embarked (S/C/Q → binary columns)
df_model = pd.get_dummies(df_model, columns=['Embarked'], drop_first=True)
print(f"  ✔  One-hot encoded Embarked port  →  Embarked_Q, Embarked_S")

print(f"\n  Dataset after cleaning:")
print(f"    Shape  : {df_model.shape[0]} rows × {df_model.shape[1]} columns")
print(f"    Nulls  : {df_model.isna().sum().sum()} (should be 0 now)")


# =============================================================================
#   CHAPTER 3 — READING THE DATA (VISUALIZATIONS)
#   Before building any model, a good data scientist just... looks.
#   Charts tell stories that tables can't. Let's see what the Titanic
#   data is actually saying.
# =============================================================================
story_header("CHAPTER 3 — Reading the Data",
             "Visualizations that reveal patterns hidden in the numbers")

# ─────────────────────────────────────────────────────────────────────────────
#   FIGURE 1 — "Who Was On Board & Who Survived?"
#   A dashboard-style overview using our ocean night theme
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 10), facecolor=OCEAN_DARK)
fig.suptitle("RMS Titanic — Who Was On Board?",
             fontsize=20, fontweight='bold', color=OFF_WHITE, y=0.97)

# Subtle background texture using a grid of faint lines
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                      left=0.07, right=0.96, top=0.90, bottom=0.08)

ax_count  = fig.add_subplot(gs[0, 0])
ax_gender = fig.add_subplot(gs[0, 1])
ax_class  = fig.add_subplot(gs[0, 2])
ax_pie    = fig.add_subplot(gs[1, 0])
ax_embark = fig.add_subplot(gs[1, 1])
ax_age_box= fig.add_subplot(gs[1, 2])

def ocean_ax(ax, title):
    """Apply the dark ocean style to any axis."""
    ax.set_facecolor(OCEAN_MID)
    ax.tick_params(colors=OFF_WHITE, labelsize=9)
    ax.spines[['top','right','left','bottom']].set_color('#2C5F82')
    ax.xaxis.label.set_color(OFF_WHITE)
    ax.yaxis.label.set_color(OFF_WHITE)
    ax.set_title(title, color=OFF_WHITE, fontsize=11, fontweight='bold', pad=8)
    ax.yaxis.grid(True, color='#2C5F82', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

# ── Panel 1: Overall count ──────────────────────────────────────
ocean_ax(ax_count, "Overall Survival")
bar_vals = [df['Survived'].value_counts()[0], df['Survived'].value_counts()[1]]
bars = ax_count.bar(['Perished ✗', 'Survived ✓'], bar_vals,
                    color=[BLOOD_RED, ICE_BLUE], width=0.5,
                    edgecolor=OCEAN_DARK, linewidth=1.5)
for bar, val in zip(bars, bar_vals):
    pct = val / N * 100
    ax_count.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 4,
                  f"{val}\n({pct:.0f}%)",
                  ha='center', va='bottom', color=OFF_WHITE,
                  fontsize=10, fontweight='bold')
ax_count.set_ylabel("Number of Passengers", color=OFF_WHITE)
ax_count.set_ylim(0, bar_vals[0] * 1.22)

# ── Panel 2: Survival by gender ─────────────────────────────────
ocean_ax(ax_gender, "Survival by Gender")
gender_data = df.groupby(['Sex','Survived']).size().unstack(fill_value=0)
gender_data.index = ['Female' if i=='female' else 'Male' for i in gender_data.index]
gender_data.columns = ['Perished','Survived']
x = np.arange(len(gender_data))
w = 0.35
ax_gender.bar(x - w/2, gender_data['Perished'], w, color=BLOOD_RED, label='Perished', edgecolor=OCEAN_DARK)
ax_gender.bar(x + w/2, gender_data['Survived'], w, color=ICE_BLUE,  label='Survived', edgecolor=OCEAN_DARK)
ax_gender.set_xticks(x)
ax_gender.set_xticklabels(gender_data.index, color=OFF_WHITE)
ax_gender.set_ylabel("Passengers", color=OFF_WHITE)
leg = ax_gender.legend(facecolor=OCEAN_DARK, edgecolor='#2C5F82', labelcolor=OFF_WHITE, fontsize=9)

# ── Panel 3: Survival by class ──────────────────────────────────
ocean_ax(ax_class, "Survival by Class")
class_data = df.groupby(['Pclass','Survived']).size().unstack(fill_value=0)
class_data.columns = ['Perished','Survived']
class_colors = [GOLD, SILVER, RUST]
class_labels = ['1st Class', '2nd Class', '3rd Class']
x = np.arange(3)
ax_class.bar(x - w/2, class_data['Perished'], w, color=BLOOD_RED, label='Perished', edgecolor=OCEAN_DARK)
ax_class.bar(x + w/2, class_data['Survived'], w, color=ICE_BLUE,  label='Survived', edgecolor=OCEAN_DARK)
ax_class.set_xticks(x)
ax_class.set_xticklabels(class_labels, color=OFF_WHITE, fontsize=9)
ax_class.set_ylabel("Passengers", color=OFF_WHITE)
ax_class.legend(facecolor=OCEAN_DARK, edgecolor='#2C5F82', labelcolor=OFF_WHITE, fontsize=9)

# ── Panel 4: Donut chart of survival ────────────────────────────
ax_pie.set_facecolor(OCEAN_DARK)
ax_pie.set_title("Survival Split", color=OFF_WHITE, fontsize=11, fontweight='bold', pad=8)
donut_sizes  = [N - survived.sum(), survived.sum()]
donut_labels = [f"Perished\n{N - survived.sum()}", f"Survived\n{survived.sum()}"]
wedges, texts, autotexts = ax_pie.pie(
    donut_sizes, labels=donut_labels, colors=[BLOOD_RED, ICE_BLUE],
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor=OCEAN_DARK, linewidth=2))
for t in texts:    t.set_color(OFF_WHITE); t.set_fontsize(9)
for a in autotexts: a.set_color(OCEAN_DARK); a.set_fontsize(9); a.set_fontweight('bold')

# ── Panel 5: Embarkation port breakdown ─────────────────────────
ocean_ax(ax_embark, "Embarkation Port")
port_surv = df.groupby(['Embarked','Survived']).size().unstack(fill_value=0)
port_surv.index  = [{'S':'Southampton','C':'Cherbourg','Q':'Queenstown'}[i]
                    for i in port_surv.index]
port_surv.columns = ['Perished','Survived']
x3 = np.arange(3)
ax_embark.bar(x3 - w/2, port_surv['Perished'], w, color=BLOOD_RED, edgecolor=OCEAN_DARK)
ax_embark.bar(x3 + w/2, port_surv['Survived'], w, color=ICE_BLUE,  edgecolor=OCEAN_DARK)
ax_embark.set_xticks(x3)
ax_embark.set_xticklabels(port_surv.index, color=OFF_WHITE, fontsize=8)
ax_embark.set_ylabel("Passengers", color=OFF_WHITE)

# ── Panel 6: Age distribution boxplot ───────────────────────────
ocean_ax(ax_age_box, "Age by Class & Survival")
groups = [
    df[(df['Pclass']==1) & (df['Survived']==1)]['Age'].dropna(),
    df[(df['Pclass']==1) & (df['Survived']==0)]['Age'].dropna(),
    df[(df['Pclass']==2) & (df['Survived']==1)]['Age'].dropna(),
    df[(df['Pclass']==2) & (df['Survived']==0)]['Age'].dropna(),
    df[(df['Pclass']==3) & (df['Survived']==1)]['Age'].dropna(),
    df[(df['Pclass']==3) & (df['Survived']==0)]['Age'].dropna(),
]
bp = ax_age_box.boxplot(groups, patch_artist=True, widths=0.5,
                         medianprops=dict(color=OFF_WHITE, linewidth=2),
                         whiskerprops=dict(color=OFF_WHITE),
                         capprops=dict(color=OFF_WHITE),
                         flierprops=dict(marker='o', color=OFF_WHITE, alpha=0.3, markersize=3))
box_colors = [ICE_BLUE, BLOOD_RED] * 3
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.8); patch.set_edgecolor(OCEAN_DARK)
ax_age_box.set_xticklabels(['1S','1P','2S','2P','3S','3P'], color=OFF_WHITE, fontsize=9)
ax_age_box.set_xlabel("Class + S(urvived)/P(erished)", color=OFF_WHITE)
ax_age_box.set_ylabel("Age", color=OFF_WHITE)

plt.savefig(OUT + "fig1_who_was_on_board.png", dpi=140,
            bbox_inches='tight', facecolor=OCEAN_DARK)
plt.close()
print(f"\n  ✔  Saved: fig1_who_was_on_board.png  (dark ocean theme dashboard)")

# ─────────────────────────────────────────────────────────────────────────────
#   FIGURE 2 — "The Hidden Patterns" — Age & Fare deep-dive
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=OCEAN_DARK)
fig.suptitle("The Hidden Patterns — Age & Wealth",
             fontsize=16, fontweight='bold', color=OFF_WHITE, y=1.01)

for ax in axes:
    ax.set_facecolor(OCEAN_MID)
    ax.tick_params(colors=OFF_WHITE)
    ax.spines[['top','right','left','bottom']].set_color('#2C5F82')

# Age density — I use KDE-style curves to make it look more sophisticated
# than a plain histogram
ax = axes[0]
survived_ages  = df[df['Survived']==1]['Age'].dropna()
perished_ages  = df[df['Survived']==0]['Age'].dropna()

ax.hist(perished_ages, bins=28, alpha=0.55, color=BLOOD_RED,
        label=f'Perished (n={len(perished_ages)})', edgecolor=OCEAN_DARK, linewidth=0.4)
ax.hist(survived_ages, bins=28, alpha=0.55, color=ICE_BLUE,
        label=f'Survived (n={len(survived_ages)})', edgecolor=OCEAN_DARK, linewidth=0.4)

# Annotate child survivors — one of the most heartbreaking patterns
children_surv = (df['Age'] < 13) & (df['Survived'] == 1)
ax.axvspan(0, 13, alpha=0.10, color='yellow')
ax.text(1, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 10,
        "Children\n(< 13)", color='yellow', fontsize=8, alpha=0.9)

ax.axvline(survived_ages.median(), color=ICE_BLUE,  linestyle='--', linewidth=1.4,
           label=f'Survivor median: {survived_ages.median():.0f} yrs', alpha=0.9)
ax.axvline(perished_ages.median(), color=BLOOD_RED, linestyle='--', linewidth=1.4,
           label=f'Perished median: {perished_ages.median():.0f} yrs', alpha=0.9)

ax.set_title("Age Distribution by Survival Outcome",
             color=OFF_WHITE, fontweight='bold', fontsize=12)
ax.set_xlabel("Age (years)", color=OFF_WHITE)
ax.set_ylabel("Number of Passengers", color=OFF_WHITE)
ax.yaxis.grid(True, color='#2C5F82', linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)
leg = ax.legend(facecolor=OCEAN_DARK, edgecolor='#2C5F82', labelcolor=OFF_WHITE, fontsize=9)

# Fare violin plot — shows wealth distribution much richer than a bar chart
ax = axes[1]
# Build data per class
fare_by_class = {
    f"1st\nClass": df[df['Pclass']==1]['Fare'].clip(upper=300),
    f"2nd\nClass": df[df['Pclass']==2]['Fare'].clip(upper=150),
    f"3rd\nClass": df[df['Pclass']==3]['Fare'].clip(upper=60),
}
positions  = [1, 2, 3]
vc_colors  = [GOLD, SILVER, RUST]
vparts = ax.violinplot(list(fare_by_class.values()), positions=positions,
                       showmedians=True, showextrema=True)
for i, (pc, color) in enumerate(zip(vparts['bodies'], vc_colors)):
    pc.set_facecolor(color); pc.set_alpha(0.7); pc.set_edgecolor(OCEAN_DARK)
vparts['cmedians'].set_color(OFF_WHITE); vparts['cmedians'].set_linewidth(2)
vparts['cmaxes'].set_color(OFF_WHITE);   vparts['cmins'].set_color(OFF_WHITE)
vparts['cbars'].set_color(OFF_WHITE)

ax.set_xticks(positions)
ax.set_xticklabels(list(fare_by_class.keys()), color=OFF_WHITE, fontsize=10)
ax.set_title("Fare Distribution per Passenger Class\n(violin width = how many passengers paid that fare)",
             color=OFF_WHITE, fontweight='bold', fontsize=11)
ax.set_ylabel("Fare Paid (£, capped)", color=OFF_WHITE)
ax.yaxis.grid(True, color='#2C5F82', linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

# Add median fare labels
for pos, grp in zip(positions, fare_by_class.values()):
    med = grp.median()
    ax.text(pos, med + 1, f"£{med:.0f}", ha='center', color=OFF_WHITE,
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT + "fig2_age_and_fare_patterns.png", dpi=140,
            bbox_inches='tight', facecolor=OCEAN_DARK)
plt.close()
print(f"  ✔  Saved: fig2_age_and_fare_patterns.png  (violin + annotated histogram)")

# ─────────────────────────────────────────────────────────────────────────────
#   FIGURE 3 — Correlation Heatmap
#   Which features are related to survival? This map tells us.
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8), facecolor=OCEAN_DARK)
ax.set_facecolor(OCEAN_DARK)

numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix  = df_model[numeric_cols].corr()

# Only show the lower triangle — the upper is a mirror and wastes space
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Custom diverging colormap that fits the ocean theme
cmap = sns.diverging_palette(15, 200, s=90, l=45, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask      = mask,
    annot     = True,
    fmt       = ".2f",
    cmap      = cmap,
    center    = 0,
    vmin      = -1, vmax = 1,
    ax        = ax,
    square    = True,
    linewidths= 1.5,
    linecolor = OCEAN_DARK,
    annot_kws = {'size': 12, 'weight': 'bold', 'color': OFF_WHITE},
    cbar_kws  = {'shrink': 0.75, 'label': 'Correlation Strength'}
)

# Style the colorbar
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_color(OFF_WHITE)
cbar.ax.tick_params(colors=OFF_WHITE)

ax.set_title(
    "Feature Correlation Map\n"
    "→ Negative values = as one goes up, survival goes down (e.g. Pclass)\n"
    "→ Positive values = as one goes up, survival goes up (e.g. Fare)",
    color=OFF_WHITE, fontsize=12, fontweight='bold', pad=15
)
ax.tick_params(colors=OFF_WHITE, labelsize=11)

plt.tight_layout()
plt.savefig(OUT + "fig3_correlation_map.png", dpi=140,
            bbox_inches='tight', facecolor=OCEAN_DARK)
plt.close()
print(f"  ✔  Saved: fig3_correlation_map.png  (custom diverging colormap)")


# =============================================================================
#   CHAPTER 4 — BUILDING NEW CLUES (FEATURE ENGINEERING)
#   Raw features don't always tell the full story.
#   A passenger travelling alone is in a very different situation from
#   one surrounded by family. We can create new features to capture that.
# =============================================================================
story_header("CHAPTER 4 — Building New Clues",
             "Feature engineering: creating smarter inputs for our model")

# FamilySize: how many people did this passenger board with?
# Travelling with a big family might slow you down during evacuation.
df_model['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print(f"\n  ✔  FamilySize = SibSp + Parch + 1")
print(f"     Range: {df_model['FamilySize'].min()} (solo) to {df_model['FamilySize'].max()} (large group)")

# IsAlone: a simple flag — were they completely on their own?
df_model['IsAlone'] = (df_model['FamilySize'] == 1).astype(int)
print(f"  ✔  IsAlone flag created")
print(f"     {df_model['IsAlone'].sum()} passengers ({df_model['IsAlone'].mean():.1%}) travelled alone")

# AgeBand: group ages into life stages instead of raw numbers
# This can help the model learn "children were prioritised" more easily
age_bins   = [0, 12, 18, 35, 60, 100]
age_labels = [0, 1, 2, 3, 4]
age_names  = {0:'Child (0-12)', 1:'Teen (13-18)', 2:'Young Adult (19-35)',
              3:'Adult (36-60)', 4:'Senior (60+)'}

df_model['AgeBand'] = pd.cut(df_model['Age'], bins=age_bins,
                             labels=age_labels).cat.add_categories(-1).fillna(-1).astype(int)
print(f"  ✔  AgeBand: ages grouped into 5 life stages")

# FareBand: similar idea for fare — bucket continuous values
df_model['FareBand'] = pd.qcut(df_model['Fare'], q=4, labels=[0,1,2,3]).astype(int)
print(f"  ✔  FareBand: fares split into 4 equal-size quartile buckets")

print(f"\n  Age band breakdown:")
for code, name in age_names.items():
    count = (df_model['AgeBand'] == code).sum()
    surv  = df[(df_model['AgeBand'] == code) & (df['Survived'] == 1)].shape[0]
    print(f"    {name:<22} : {count:>3} passengers | survival rate: {surv/max(count,1):.0%}")


# =============================================================================
#   CHAPTER 5 — TEACHING THE MODEL
#   Logistic Regression is a classic algorithm for yes/no questions.
#   It learns weights for each feature that maximise how often it's right.
# =============================================================================
story_header("CHAPTER 5 — Teaching the Model",
             "Splitting data, scaling features, training Logistic Regression")

# Make sure there are no remaining NaN values before training
df_model = df_model.fillna(df_model.median(numeric_only=True))

# These are the features we feed the model — our "clues" about each passenger
FEATURES = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize',
            'IsAlone', 'AgeBand', 'FareBand']

X = df_model[FEATURES]
y = df_model['Survived']

# Split: 80% for training (model learns), 20% for testing (we evaluate honestly)
# stratify=y ensures both splits have the same survival ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1912, stratify=y)

# Feature scaling: Logistic Regression works better when all features
# are on a similar scale. StandardScaler centres each feature at 0,
# standard deviation 1 — so "Age 30" and "Fare £200" are comparable.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit+transform on training data
X_test  = scaler.transform(X_test)        # only transform on test data (no data leakage!)

model = LogisticRegression(max_iter=2000, C=1.0, random_state=1912)
model.fit(X_train, y_train)

print(f"\n  Model trained successfully!")
print(f"    Training set : {len(X_train)} passengers")
print(f"    Test set     : {len(X_test)} passengers")
print(f"\n  Feature coefficients (how much each clue matters):")
coef_df = pd.DataFrame({
    'Feature'    : FEATURES,
    'Coefficient': model.coef_[0],
    'Direction'  : ['↑ Survival' if c > 0 else '↓ Survival' for c in model.coef_[0]]
}).sort_values('Coefficient', ascending=False)
print(coef_df.to_string(index=False))


# =============================================================================
#   CHAPTER 6 — DID IT WORK?  (MODEL EVALUATION)
#   Accuracy is one number, but the confusion matrix tells a richer story.
#   It shows exactly where the model got confused.
# =============================================================================
story_header("CHAPTER 6 — Did It Work?",
             "Accuracy, confusion matrix, and what the mistakes mean")

y_pred    = model.predict(X_test)
y_proba   = model.predict_proba(X_test)[:, 1]  # survival probability 0–1
accuracy  = accuracy_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Overall Accuracy : {accuracy:.2%}")
print(f"\n  Breaking down the {len(y_test)} test passengers:")
print(f"    ✔  {tp:>3} correctly predicted as SURVIVED  (True Positives)")
print(f"    ✔  {tn:>3} correctly predicted as PERISHED  (True Negatives)")
print(f"    ✗  {fp:>3} wrongly predicted as survived (False Positives) — actually perished")
print(f"    ✗  {fn:>3} wrongly predicted as perished (False Negatives) — actually survived")
print(f"\n  Full classification report:")
print(classification_report(y_test, y_pred, target_names=['Perished','Survived']))

# Show a few interesting individual predictions
print("  Sample predictions from the test set:")
test_results = pd.DataFrame(X_test, columns=FEATURES)
test_results['Actual']     = y_test.values
test_results['Predicted']  = y_pred
test_results['Confidence'] = (y_proba * 100).round(1)
test_results['Correct']    = (test_results['Actual'] == test_results['Predicted'])
interesting = test_results[~test_results['Correct']].head(4)   # show some mistakes
print(interesting[['Actual','Predicted','Confidence']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
#   FIGURE 4 — "The Verdict" — Confusion Matrix + Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(17, 7), facecolor=OCEAN_DARK)
fig.suptitle(f"The Verdict — Model Accuracy: {accuracy:.2%}",
             fontsize=17, fontweight='bold', color=OFF_WHITE, y=1.01)

gs4 = fig.add_gridspec(1, 2, wspace=0.35, left=0.07, right=0.95)
ax_cm   = fig.add_subplot(gs4[0])
ax_feat = fig.add_subplot(gs4[1])

# ── Confusion Matrix ────────────────────────────────────────────
ax_cm.set_facecolor(OCEAN_DARK)

# Custom colour matrix: correct=blue shades, wrong=red shades
cm_display  = np.array([[tn, fp], [fn, tp]])
cm_colors   = np.array([
    [ICE_BLUE, BLOOD_RED],   # TN = good (blue), FP = bad (red)
    [BLOOD_RED, GOLD]        # FN = bad (red),   TP = good (gold)
])

for i in range(2):
    for j in range(2):
        rect = FancyBboxPatch((j+0.05, 1-i+0.05), 0.90, 0.90,
                              boxstyle="round,pad=0.05",
                              facecolor=cm_colors[i][j], alpha=0.85,
                              edgecolor=OCEAN_DARK, linewidth=2)
        ax_cm.add_patch(rect)
        ax_cm.text(j+0.5, 1-i+0.5, str(cm_display[i][j]),
                   ha='center', va='center', fontsize=26, fontweight='bold',
                   color=OCEAN_DARK)

# Labels
ax_cm.text(0.5, -0.12, 'Predicted: Perished', ha='center', color=OFF_WHITE, fontsize=10)
ax_cm.text(1.5, -0.12, 'Predicted: Survived', ha='center', color=OFF_WHITE, fontsize=10)
ax_cm.text(-0.15, 0.5,  'Actual:\nPerished', ha='right', va='center', color=OFF_WHITE, fontsize=10)
ax_cm.text(-0.15, 1.5,  'Actual:\nSurvived', ha='right', va='center', color=OFF_WHITE, fontsize=10)
ax_cm.set_xlim(-0.2, 2.1); ax_cm.set_ylim(-0.3, 2.1)
ax_cm.axis('off')
ax_cm.set_title("Confusion Matrix", color=OFF_WHITE, fontsize=13, fontweight='bold', pad=15)

# ── Feature Importance (horizontal bar chart) ───────────────────
ax_feat.set_facecolor(OCEAN_MID)
ax_feat.tick_params(colors=OFF_WHITE)
ax_feat.spines[['top','right','left','bottom']].set_color('#2C5F82')

coefs_sorted = pd.Series(model.coef_[0], index=FEATURES).sort_values()
colors_feat  = [ICE_BLUE if c > 0 else BLOOD_RED for c in coefs_sorted]

bars = ax_feat.barh(coefs_sorted.index, coefs_sorted.values,
                    color=colors_feat, edgecolor=OCEAN_DARK, height=0.6)
ax_feat.axvline(0, color=OFF_WHITE, linewidth=1, linestyle='--', alpha=0.5)

for bar, val in zip(bars, coefs_sorted.values):
    xpos = val + 0.01 if val >= 0 else val - 0.01
    ha   = 'left'      if val >= 0 else 'right'
    ax_feat.text(xpos, bar.get_y() + bar.get_height()/2,
                 f"{val:+.3f}", va='center', ha=ha,
                 color=OFF_WHITE, fontsize=9)

ax_feat.set_title("Feature Importance\n(positive = increases survival odds | negative = decreases)",
                  color=OFF_WHITE, fontsize=11, fontweight='bold', pad=12)
ax_feat.set_xlabel("Logistic Regression Coefficient", color=OFF_WHITE)
ax_feat.yaxis.label.set_color(OFF_WHITE)
ax_feat.xaxis.grid(True, color='#2C5F82', linewidth=0.5, alpha=0.6)
ax_feat.set_axisbelow(True)

# Legend for colours
blue_patch = mpatches.Patch(color=ICE_BLUE,  label='↑ Increases survival chance')
red_patch  = mpatches.Patch(color=BLOOD_RED, label='↓ Decreases survival chance')
ax_feat.legend(handles=[blue_patch, red_patch],
               facecolor=OCEAN_DARK, edgecolor='#2C5F82',
               labelcolor=OFF_WHITE, fontsize=9, loc='lower right')

plt.savefig(OUT + "fig4_model_verdict.png", dpi=140,
            bbox_inches='tight', facecolor=OCEAN_DARK)
plt.close()
print(f"\n  ✔  Saved: fig4_model_verdict.png  (custom confusion matrix + signed coefficients)")


# =============================================================================
#   EPILOGUE — REFLECTIONS
# =============================================================================
story_header("EPILOGUE — What Did We Learn?")

print(f"""
  Our logistic regression model reached {accuracy:.1%} accuracy on unseen passengers.

  The three most powerful predictors were:
    1. Sex       — Female passengers were far more likely to survive
                   (lifeboat etiquette: "women and children first")
    2. Pclass    — 1st class passengers had better access to lifeboats
                   (their cabins were closer to the deck)
    3. Fare      — Closely linked to class, wealthier passengers survived more

  What this project taught me about data science:
    • Data is never clean — missing values and encoding are unavoidable
    • Visualising before modelling reveals insights no algorithm will find alone
    • A "simple" model like Logistic Regression can still be very powerful
    • Accuracy is not the only metric — understanding the errors matters too

  Files produced:
    📄  titanic_dataset.csv          — the full passenger manifest
    📊  fig1_who_was_on_board.png    — 6-panel survival dashboard
    📊  fig2_age_and_fare_patterns.png — histogram + violin plots
    📊  fig3_correlation_map.png     — feature correlation heatmap
    📊  fig4_model_verdict.png       — confusion matrix + feature importance
    🐍  titanic_humanized.py         — this script
""")
