"""
MindMatch – Data Generation
============================
STEP 1: Load real DASS-21/Student-Stress data  (or synthesise if absent)
         → realistic depression / anxiety / stress profiles for 1,000 users
STEP 2: Generate synthetic activity ratings
         → simulate user–activity interactions  (≥5,000 total)
STEP 3: Combine and save
         → products.csv | users.csv | ratings.csv

If you have a Kaggle dataset, convert it first:
    python backend/data/prepare_dass21.py <path_to_csv>
Then restart the server – it picks up dass21_raw.csv automatically.
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# DASS-42-equivalent severity cutoffs  (scores 0–42)
# ---------------------------------------------------------------------------
def _dass_severity(score, dim):
    score = int(score)
    if dim == "depression":
        if score <= 9:   return "Normal"
        if score <= 13:  return "Mild"
        if score <= 20:  return "Moderate"
        if score <= 27:  return "Severe"
        return "Extremely Severe"
    elif dim == "anxiety":
        if score <= 7:   return "Normal"
        if score <= 9:   return "Mild"
        if score <= 14:  return "Moderate"
        if score <= 19:  return "Severe"
        return "Extremely Severe"
    else:  # stress
        if score <= 14:  return "Normal"
        if score <= 18:  return "Mild"
        if score <= 25:  return "Moderate"
        if score <= 33:  return "Severe"
        return "Extremely Severe"


# ---------------------------------------------------------------------------
# DASS-21 raw CSV loader  (4 supported Kaggle formats)
# ---------------------------------------------------------------------------
_DEPRESSION_ITEMS = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
_ANXIETY_ITEMS    = [2, 4, 7,  9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
_STRESS_ITEMS     = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]


def _load_dass21_csv(data_dir):
    """
    Load real DASS data from dass21_raw.csv.
    Returns DataFrame(dass_depression, dass_anxiety, dass_stress) or None.

    Supported formats:
      (a) depression, anxiety, stress              (0-42 already)
      (b) dass_depression, dass_anxiety, dass_stress
      (c) Q1A … Q42A                               (raw DASS-42 items, values 1-4 or 0-3)
      (d) anxiety_level, depression, stress_level  (Student Stress Factors, 0-21 → ×2)
    """
    path = os.path.join(data_dir, "dass21_raw.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path).dropna(how="all")
        cols = set(df.columns)

        # (a)
        if {"depression", "anxiety", "stress"}.issubset(cols):
            out = df[["depression", "anxiety", "stress"]].dropna().astype(int).clip(0, 42)
            out.columns = ["dass_depression", "dass_anxiety", "dass_stress"]

        # (b)
        elif {"dass_depression", "dass_anxiety", "dass_stress"}.issubset(cols):
            out = df[["dass_depression", "dass_anxiety", "dass_stress"]].dropna().astype(int).clip(0, 42)

        # (c) raw Q1A-Q42A items
        elif "Q1A" in cols and "Q42A" in cols:
            q_available = [f"Q{i}A" for i in range(1, 43) if f"Q{i}A" in cols]
            max_val = df[q_available].max().max()
            offset  = 1 if max_val > 3 else 0  # recode 1-4 → 0-3

            def _item_sum(row, items):
                return int(np.clip(
                    sum(max(0, int(row.get(f"Q{i}A", 0)) - offset) for i in items), 0, 42
                ))

            rows = [
                {
                    "dass_depression": _item_sum(r, _DEPRESSION_ITEMS),
                    "dass_anxiety":    _item_sum(r, _ANXIETY_ITEMS),
                    "dass_stress":     _item_sum(r, _STRESS_ITEMS),
                }
                for _, r in df.iterrows()
            ]
            out = pd.DataFrame(rows)

        # (d) Student Stress Factors (0-21 scale → ×2)
        elif {"anxiety_level", "depression", "stress_level"}.issubset(cols):
            out = pd.DataFrame({
                "dass_depression": (df["depression"]   * 2).clip(0, 42).astype(int),
                "dass_anxiety":    (df["anxiety_level"]* 2).clip(0, 42).astype(int),
                "dass_stress":     (df["stress_level"] * 2).clip(0, 42).astype(int),
            })

        else:
            print("  dass21_raw.csv found but columns not recognised – using synthetic profiles.")
            print("  Run: python backend/data/prepare_dass21.py <your_kaggle_csv>")
            return None

        out = out[(out.sum(axis=1)) > 0].reset_index(drop=True)
        print(f"  STEP 1 complete – loaded {len(out)} real DASS records from dass21_raw.csv")
        return out

    except Exception as e:
        print(f"  Could not parse dass21_raw.csv ({e}) – using synthetic profiles.")
    return None


# ---------------------------------------------------------------------------
# Main data generation
# ---------------------------------------------------------------------------
def generate_data(n_users=1000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # ==========================================================================
    # Activities catalogue (60 wellness activities)
    # ==========================================================================
    activities = [
        # ---- Mindfulness & Meditation (A001–A012) ----
        {"product_id": "A001", "name": "Guided Body Scan Meditation",    "category": "Mindfulness", "duration_minutes": 20, "stress_target": "stress",     "effectiveness": "high",     "description": "Full body awareness meditation reduces physical tension and mental fatigue",                "tags": "meditation mindfulness body relaxation tension stress relief calm focus"},
        {"product_id": "A002", "name": "Mindful Breathing Exercise",     "category": "Mindfulness", "duration_minutes": 10, "stress_target": "anxiety",    "effectiveness": "high",     "description": "Diaphragmatic breathing activates parasympathetic nervous system to calm anxiety",         "tags": "breathing mindfulness anxiety calm nervous quick relief diaphragm"},
        {"product_id": "A003", "name": "Progressive Muscle Relaxation",  "category": "Mindfulness", "duration_minutes": 25, "stress_target": "stress",     "effectiveness": "high",     "description": "Systematically tense and release muscle groups to reduce physical stress response",          "tags": "muscle relaxation stress physical tension body release pmr progressive"},
        {"product_id": "A004", "name": "Loving-Kindness Meditation",     "category": "Mindfulness", "duration_minutes": 20, "stress_target": "depression", "effectiveness": "high",     "description": "Cultivate compassion and positive emotions toward self and others",                         "tags": "meditation compassion kindness depression positivity self love warm"},
        {"product_id": "A005", "name": "Mindfulness App Session",        "category": "Mindfulness", "duration_minutes": 15, "stress_target": "all",        "effectiveness": "moderate", "description": "Guided mindfulness session using a digital app suitable for all beginners",                 "tags": "mindfulness app guided beginner meditation digital calm daily all"},
        {"product_id": "A006", "name": "4-7-8 Breathing Technique",      "category": "Mindfulness", "duration_minutes": 10, "stress_target": "anxiety",    "effectiveness": "high",     "description": "Structured breathing pattern that quickly reduces anxiety and promotes sleep onset",         "tags": "breathing anxiety sleep quick technique nervous calm 478 structure"},
        {"product_id": "A007", "name": "Grounding 5-4-3-2-1 Technique",  "category": "Mindfulness", "duration_minutes": 10, "stress_target": "anxiety",    "effectiveness": "high",     "description": "Sensory grounding exercise to interrupt anxiety spirals and panic attacks",                  "tags": "grounding anxiety panic sensory present moment awareness technique"},
        {"product_id": "A008", "name": "Morning Mindfulness Routine",    "category": "Mindfulness", "duration_minutes": 30, "stress_target": "stress",     "effectiveness": "high",     "description": "Start your day with intention through meditation breathing and gratitude practice",           "tags": "morning routine stress meditation gratitude intention daily habit"},
        {"product_id": "A009", "name": "Mindful Walking",                "category": "Mindfulness", "duration_minutes": 20, "stress_target": "all",        "effectiveness": "moderate", "description": "Walk with full sensory awareness integrating movement and meditation practice",                "tags": "walking mindfulness movement outdoor awareness nature calm all gentle"},
        {"product_id": "A010", "name": "Zen Coloring / Mandala",         "category": "Mindfulness", "duration_minutes": 30, "stress_target": "anxiety",    "effectiveness": "moderate", "description": "Meditative coloring to focus the mind and quiet anxious overthinking",                       "tags": "coloring mandala zen anxiety focus creative calming art pattern"},
        {"product_id": "A011", "name": "Yoga Nidra (Sleep Yoga)",        "category": "Mindfulness", "duration_minutes": 45, "stress_target": "stress",     "effectiveness": "high",     "description": "Deep guided relaxation at the threshold between wakefulness and sleep",                      "tags": "yoga nidra sleep stress deep relaxation guided rest theta"},
        {"product_id": "A012", "name": "Visualization Meditation",       "category": "Mindfulness", "duration_minutes": 20, "stress_target": "depression", "effectiveness": "moderate", "description": "Guided imagery to create positive mental experiences and elevate mood",                       "tags": "visualization meditation depression mood positive imagery mental uplift"},
        # ---- Physical Activity (A013–A022) ----
        {"product_id": "A013", "name": "Yoga (Beginner)",                "category": "Physical",    "duration_minutes": 30, "stress_target": "all",        "effectiveness": "high",     "description": "Gentle yoga sequences combining postures breathwork and mindfulness for all levels",         "tags": "yoga beginner posture breathing flexibility mindfulness all gentle"},
        {"product_id": "A014", "name": "Yoga (Intermediate)",            "category": "Physical",    "duration_minutes": 45, "stress_target": "stress",     "effectiveness": "high",     "description": "Dynamic yoga flow building strength flexibility and mental resilience",                       "tags": "yoga intermediate flow strength flexibility stress mental resilience"},
        {"product_id": "A015", "name": "Morning Walk / Jog",             "category": "Physical",    "duration_minutes": 30, "stress_target": "depression", "effectiveness": "high",     "description": "Aerobic exercise releases endorphins and serotonin to directly combat depression",           "tags": "walk jog exercise depression endorphin serotonin outdoor morning aerobic"},
        {"product_id": "A016", "name": "Stretching Routine",             "category": "Physical",    "duration_minutes": 15, "stress_target": "stress",     "effectiveness": "moderate", "description": "Full-body stretching releases accumulated muscle tension and improves body awareness",        "tags": "stretching stress tension body flexibility quick daily release"},
        {"product_id": "A017", "name": "Dance Exercise / Zumba",         "category": "Physical",    "duration_minutes": 45, "stress_target": "depression", "effectiveness": "high",     "description": "Upbeat dancing combining cardio and joy to powerfully boost mood and energy levels",         "tags": "dance zumba depression mood cardio fun energy upbeat rhythm joy"},
        {"product_id": "A018", "name": "Swimming",                       "category": "Physical",    "duration_minutes": 45, "stress_target": "stress",     "effectiveness": "high",     "description": "Low-impact full-body exercise with meditative water effects for profound stress relief",      "tags": "swimming stress water exercise meditative low impact body float"},
        {"product_id": "A019", "name": "Cycling Outdoors",               "category": "Physical",    "duration_minutes": 45, "stress_target": "depression", "effectiveness": "moderate", "description": "Outdoor cycling combining physical activity with nature exposure and freedom",                "tags": "cycling outdoor exercise depression nature exposure cardio freedom"},
        {"product_id": "A020", "name": "HIIT Workout",                   "category": "Physical",    "duration_minutes": 30, "stress_target": "stress",     "effectiveness": "high",     "description": "High-intensity interval training burns excess stress hormones and dramatically boosts mood",  "tags": "hiit workout stress cortisol burn intense cardio mood interval"},
        {"product_id": "A021", "name": "Nature Hike",                    "category": "Physical",    "duration_minutes": 90, "stress_target": "all",        "effectiveness": "high",     "description": "Immersive nature experience combining aerobic exercise with forest therapy benefits",         "tags": "hike nature outdoor exercise forest therapy all mood restoration"},
        {"product_id": "A022", "name": "Tai Chi / Qigong",               "category": "Physical",    "duration_minutes": 30, "stress_target": "anxiety",    "effectiveness": "high",     "description": "Slow meditative movements balance energy and calm the overactivated nervous system",          "tags": "tai chi qigong anxiety meditation movement slow energy calm balance"},
        # ---- Creative Expression (A023–A030) ----
        {"product_id": "A023", "name": "Gratitude Journaling",           "category": "Creative",    "duration_minutes": 15, "stress_target": "depression", "effectiveness": "high",     "description": "Write three things you are grateful for to rewire persistent negative thought patterns",    "tags": "journaling gratitude depression positive thinking rewire habit daily"},
        {"product_id": "A024", "name": "Art Therapy",                    "category": "Creative",    "duration_minutes": 45, "stress_target": "anxiety",    "effectiveness": "high",     "description": "Expressive art making externalizes anxiety and promotes healthy emotional processing",        "tags": "art therapy drawing painting anxiety expression emotion creative visual"},
        {"product_id": "A025", "name": "Music Listening (Playlist)",     "category": "Creative",    "duration_minutes": 30, "stress_target": "all",        "effectiveness": "moderate", "description": "Curated calming or uplifting music to regulate mood and lower stress arousal",              "tags": "music playlist calming uplifting mood stress regulation emotion all"},
        {"product_id": "A026", "name": "Creative Writing",               "category": "Creative",    "duration_minutes": 45, "stress_target": "depression", "effectiveness": "moderate", "description": "Storytelling and narrative writing to process emotions and reconstruct meaning",              "tags": "writing creative depression storytelling narrative emotion meaning process"},
        {"product_id": "A027", "name": "Photography Walk",               "category": "Creative",    "duration_minutes": 60, "stress_target": "all",        "effectiveness": "moderate", "description": "Mindful photography combines outdoor movement with creative focus and presence",              "tags": "photography walk outdoor creative mindful focus nature all presence"},
        {"product_id": "A028", "name": "DIY Craft Project",              "category": "Creative",    "duration_minutes": 60, "stress_target": "anxiety",    "effectiveness": "moderate", "description": "Hands-on creative work produces flow state that interrupts anxious rumination",              "tags": "craft diy anxiety hands creative flow state rumination make"},
        {"product_id": "A029", "name": "Playing Musical Instrument",     "category": "Creative",    "duration_minutes": 30, "stress_target": "stress",     "effectiveness": "high",     "description": "Active music making demands full attention providing a powerful stress relief valve",         "tags": "music instrument stress relief creative engagement attention focus play"},
        {"product_id": "A030", "name": "Cooking a New Recipe",           "category": "Creative",    "duration_minutes": 60, "stress_target": "depression", "effectiveness": "moderate", "description": "Mindful cooking engages all senses and delivers accomplishment to lift low mood",             "tags": "cooking recipe depression mindful sensory accomplishment mood new"},
        # ---- Social Connection (A031–A038) ----
        {"product_id": "A031", "name": "Call a Friend / Family",         "category": "Social",      "duration_minutes": 20, "stress_target": "depression", "effectiveness": "high",     "description": "Social connection and verbal sharing is one of the most evidence-based antidepressants",    "tags": "social call friend family depression connection support sharing voice"},
        {"product_id": "A032", "name": "Group Support Session",          "category": "Social",      "duration_minutes": 60, "stress_target": "all",        "effectiveness": "high",     "description": "Structured peer support group providing shared experience validation and belonging",          "tags": "support group peer connection shared experience all therapy belonging"},
        {"product_id": "A033", "name": "Volunteer Community Work",       "category": "Social",      "duration_minutes": 120,"stress_target": "depression", "effectiveness": "high",     "description": "Helping others creates purpose and meaning which strongly counteracts depression",            "tags": "volunteer community depression purpose meaning helping others give"},
        {"product_id": "A034", "name": "Family Game Night",              "category": "Social",      "duration_minutes": 90, "stress_target": "depression", "effectiveness": "moderate", "description": "Light-hearted family bonding through play to reduce isolation and naturally boost mood",      "tags": "family game play depression bonding isolation mood fun laughter"},
        {"product_id": "A035", "name": "Join a Club or Group",           "category": "Social",      "duration_minutes": 90, "stress_target": "depression", "effectiveness": "moderate", "description": "Structured social activity with shared interests to build belonging and reduce loneliness",   "tags": "club group social depression belonging interest community join"},
        {"product_id": "A036", "name": "Coffee Chat with Colleague",     "category": "Social",      "duration_minutes": 30, "stress_target": "stress",     "effectiveness": "moderate", "description": "Casual workplace social interaction helps process daily stress and build resilience",          "tags": "social work colleague stress casual chat connection resilience workplace"},
        {"product_id": "A037", "name": "Attend a Social Event",          "category": "Social",      "duration_minutes": 120,"stress_target": "depression", "effectiveness": "moderate", "description": "Social engagement and new experiences actively combat isolation and depressive withdrawal",    "tags": "social event depression isolation engagement experience new community"},
        {"product_id": "A038", "name": "Online Community Forum",         "category": "Social",      "duration_minutes": 30, "stress_target": "depression", "effectiveness": "low",      "description": "Virtual connection through shared interest communities for accessible mild social support",   "tags": "online forum community depression virtual connection social support mild"},
        # ---- Rest & Recovery (A039–A046) ----
        {"product_id": "A039", "name": "Power Nap (20 min)",             "category": "Rest",        "duration_minutes": 20, "stress_target": "stress",     "effectiveness": "high",     "description": "Short restorative nap proven to reduce cortisol levels and restore cognitive function",      "tags": "nap sleep stress cortisol rest restore cognitive short recovery power"},
        {"product_id": "A040", "name": "Digital Detox Hour",             "category": "Rest",        "duration_minutes": 60, "stress_target": "stress",     "effectiveness": "high",     "description": "Screen-free time breaks the chronic stress feedback loop of constant digital stimulation",   "tags": "digital detox screen stress break phone offline rest calm unplug"},
        {"product_id": "A041", "name": "Reading Fiction",                "category": "Rest",        "duration_minutes": 45, "stress_target": "anxiety",    "effectiveness": "moderate", "description": "Engaging fiction provides cognitive escape and has been shown to reduce anxiety by 68%",     "tags": "reading fiction anxiety escape cognitive rest relaxation calm immerse"},
        {"product_id": "A042", "name": "Hot Bath / Shower Relaxation",   "category": "Rest",        "duration_minutes": 20, "stress_target": "stress",     "effectiveness": "high",     "description": "Warm water lowers cortisol prepares the body for rest and powerfully reduces stress",        "tags": "bath shower warm water stress cortisol relax rest recovery soak"},
        {"product_id": "A043", "name": "Forest Bathing",                 "category": "Rest",        "duration_minutes": 60, "stress_target": "anxiety",    "effectiveness": "high",     "description": "Shinrin-yoku lowers cortisol blood pressure and anxiety through deep nature immersion",      "tags": "forest bathing nature anxiety cortisol blood pressure shinrin calm trees"},
        {"product_id": "A044", "name": "Aromatherapy Session",           "category": "Rest",        "duration_minutes": 20, "stress_target": "anxiety",    "effectiveness": "moderate", "description": "Lavender and chamomile scents activate the relaxation response to reduce anxiety",           "tags": "aromatherapy lavender chamomile anxiety relax scent calm sensory oils"},
        {"product_id": "A045", "name": "Gentle Stretching Before Bed",   "category": "Rest",        "duration_minutes": 15, "stress_target": "stress",     "effectiveness": "moderate", "description": "Light evening stretching signals the body to wind down and improves sleep quality",         "tags": "stretching evening sleep stress wind down bedtime gentle slow"},
        {"product_id": "A046", "name": "Sleep Hygiene Routine",          "category": "Rest",        "duration_minutes": 30, "stress_target": "stress",     "effectiveness": "high",     "description": "Consistent pre-sleep routine regulates circadian rhythm and reduces stress-induced insomnia", "tags": "sleep hygiene routine stress insomnia circadian bedtime consistent wind"},
        # ---- Cognitive Techniques (A047–A056) ----
        {"product_id": "A047", "name": "CBT Journaling",                 "category": "Cognitive",   "duration_minutes": 30, "stress_target": "depression", "effectiveness": "high",     "description": "Cognitive behavioral journaling to identify and challenge negative automatic thoughts",       "tags": "cbt journaling depression cognitive behavioral negative thoughts challenge"},
        {"product_id": "A048", "name": "Problem-Solving Workshop",       "category": "Cognitive",   "duration_minutes": 45, "stress_target": "stress",     "effectiveness": "moderate", "description": "Structured problem-solving technique to reduce helplessness and cognitive overwhelm",         "tags": "problem solving stress helpless overwhelm technique structured step"},
        {"product_id": "A049", "name": "Time Management Planning",       "category": "Cognitive",   "duration_minutes": 30, "stress_target": "stress",     "effectiveness": "moderate", "description": "Prioritization and time-blocking techniques to reduce overwhelm and restore control",          "tags": "time management stress schedule priority planning overwhelm control"},
        {"product_id": "A050", "name": "Positive Affirmations",          "category": "Cognitive",   "duration_minutes": 10, "stress_target": "depression", "effectiveness": "moderate", "description": "Daily positive self-statements to counter negative self-talk and improve self-worth",        "tags": "affirmations depression positive self-talk worth daily mindset belief"},
        {"product_id": "A051", "name": "Worry Time Scheduling",          "category": "Cognitive",   "duration_minutes": 20, "stress_target": "anxiety",    "effectiveness": "high",     "description": "Contain anxiety by scheduling a specific daily worry window to break chronic rumination",     "tags": "worry anxiety rumination schedule contain limit cbt technique window"},
        {"product_id": "A052", "name": "Thought Record Exercise",        "category": "Cognitive",   "duration_minutes": 30, "stress_target": "anxiety",    "effectiveness": "high",     "description": "Identify and restructure catastrophic thinking patterns that drive chronic anxiety",          "tags": "thought record anxiety catastrophic thinking cbt restructure pattern"},
        {"product_id": "A053", "name": "Values Clarification",           "category": "Cognitive",   "duration_minutes": 45, "stress_target": "depression", "effectiveness": "moderate", "description": "Identify core personal values to restore sense of meaning purpose and life direction",        "tags": "values depression meaning purpose direction life clarity core"},
        {"product_id": "A054", "name": "Behavioral Activation",          "category": "Cognitive",   "duration_minutes": 30, "stress_target": "depression", "effectiveness": "high",     "description": "Schedule rewarding activities to systematically break the inactivity-depression cycle",      "tags": "behavioral activation depression activity schedule reward cycle break"},
        {"product_id": "A055", "name": "Goal Setting Session",           "category": "Cognitive",   "duration_minutes": 45, "stress_target": "all",        "effectiveness": "moderate", "description": "SMART goal setting to rebuild motivation confidence and sense of accomplishment",             "tags": "goal setting smart motivation confidence accomplishment all rebuild"},
        {"product_id": "A056", "name": "Self-Compassion Exercise",       "category": "Cognitive",   "duration_minutes": 20, "stress_target": "depression", "effectiveness": "high",     "description": "Practice treating yourself with the same kindness you would offer a struggling friend",     "tags": "self compassion depression kindness inner critic mindfulness friend"},
        # ---- Lifestyle & Wellness (A057–A060) ----
        {"product_id": "A057", "name": "Healthy Meal Prep",              "category": "Lifestyle",   "duration_minutes": 60, "stress_target": "all",        "effectiveness": "moderate", "description": "Prepare nutritious meals to support brain chemistry and stabilize mood through nutrition",    "tags": "nutrition meal prep health mood brain chemistry lifestyle all omega"},
        {"product_id": "A058", "name": "Herbal Tea Ritual",              "category": "Lifestyle",   "duration_minutes": 15, "stress_target": "anxiety",    "effectiveness": "moderate", "description": "Chamomile or passionflower tea ritual to calm the nervous system and prepare for rest",       "tags": "herbal tea chamomile anxiety nervous system calm ritual bedtime plant"},
        {"product_id": "A059", "name": "Sunlight Exposure Walk",         "category": "Lifestyle",   "duration_minutes": 20, "stress_target": "depression", "effectiveness": "high",     "description": "Morning sunlight regulates serotonin and melatonin production fighting seasonal depression",  "tags": "sunlight morning depression serotonin melatonin seasonal light vitamin"},
        {"product_id": "A060", "name": "Free-Form Journaling",           "category": "Lifestyle",   "duration_minutes": 20, "stress_target": "all",        "effectiveness": "moderate", "description": "Unstructured expressive writing to process thoughts emotions and daily experiences",         "tags": "journaling writing emotion thoughts expression daily all process free"},
    ]

    products_df = pd.DataFrame(activities)
    products_df["price"] = products_df["duration_minutes"]  # alias for model compatibility

    # ==========================================================================
    # STEP 1 — User profiles with DASS-21 stress scores
    # ==========================================================================
    first_names = [
        "Alice","Bob","Charlie","Diana","Eve","Frank","Grace","Henry","Iris","Jack",
        "Kate","Liam","Mia","Noah","Olivia","Peter","Quinn","Rachel","Sam","Tina",
        "Uma","Victor","Wendy","Xavier","Yara","Zoe","Aaron","Beth","Carl","Dana",
        "Ethan","Fiona","George","Hannah","Ian","Julia","Kyle","Laura","Mark","Nancy",
        "Oscar","Paula","Ryan","Sara","Tom","Una","Vince","Willa","Xander","Yvonne",
        "Zack","Amber","Brian","Clara","Derek","Emily","Felix","Gina","Hugo","Isabel",
        "James","Karen","Leo","Monica","Nate","Penny","Paul","Rita","Steve","Tara",
        "Ulric","Vera","Will","Xena","Yasmin","Zara","Alex","Bella","Chris","Daisy",
        "Elliot","Freya","Gavin","Holly","Ivan","Jade","Kirk","Luna","Max","Nina",
        "Oliver","Polly","Rex","Stella","Tyler","Ursula","Val","Wade","Xia","Yuki",
    ]

    occupation_pool = [
        "Student", "Student", "Student",
        "Professional", "Professional",
        "Healthcare Worker", "Teacher",
        "Artist", "Freelancer", "Researcher",
    ]
    age_pool = ["18-25", "18-25", "26-35", "26-35", "36-50", "51+"]

    # STEP 1: try real DASS data first
    real_dass = _load_dass21_csv(data_dir)
    if real_dass is None:
        print("  STEP 1 – No dass21_raw.csv found. Generating synthetic DASS profiles.")
        print("  Tip: download a DASS dataset from Kaggle and run prepare_dass21.py")

    users = []
    for i in range(1, n_users + 1):
        # Name: cycle through pool, append suffix for users beyond pool size
        base_name = first_names[(i - 1) % len(first_names)]
        suffix    = f"_{(i - 1) // len(first_names) + 1}" if i > len(first_names) else ""
        name      = base_name + suffix

        if real_dass is not None and (i - 1) < len(real_dass):
            row   = real_dass.iloc[i - 1]
            dep   = int(row["dass_depression"])
            anx   = int(row["dass_anxiety"])
            str_s = int(row["dass_stress"])
        else:
            # Synthetic — realistic population distribution
            profile_type = np.random.choice(
                ["low", "mild", "moderate", "high", "severe"],
                p=[0.30, 0.25, 0.25, 0.15, 0.05],
            )
            params = {
                "low":      {"dep": (4, 2),  "anx": (3, 2),  "str": (8,  3)},
                "mild":     {"dep": (11, 2), "anx": (8, 1),  "str": (16, 2)},
                "moderate": {"dep": (17, 2), "anx": (12, 2), "str": (22, 3)},
                "high":     {"dep": (23, 2), "anx": (16, 2), "str": (29, 3)},
                "severe":   {"dep": (32, 3), "anx": (21, 2), "str": (36, 3)},
            }[profile_type]
            dep   = int(np.clip(round(np.random.normal(*params["dep"])),  0, 42))
            anx   = int(np.clip(round(np.random.normal(*params["anx"])),  0, 42))
            str_s = int(np.clip(round(np.random.normal(*params["str"])),  0, 42))

        users.append({
            "user_id":             f"U{i:04d}",
            "name":                name,
            "age_group":           random.choice(age_pool),
            "occupation":          random.choice(occupation_pool),
            "dass_depression":     dep,
            "dass_anxiety":        anx,
            "dass_stress":         str_s,
            "depression_severity": _dass_severity(dep,   "depression"),
            "anxiety_severity":    _dass_severity(anx,   "anxiety"),
            "stress_severity":     _dass_severity(str_s, "stress"),
            "sleep_quality":       round(float(np.clip(np.random.normal(3.0, 0.8), 1, 5)), 1),
            "social_support":      round(float(np.clip(np.random.normal(3.2, 0.9), 1, 5)), 1),
            "dietary_preference":  "none",
        })

    users_df = pd.DataFrame(users)
    print(f"  STEP 1 complete – {len(users_df)} user profiles created.")

    # ==========================================================================
    # STEP 2 — Synthetic activity ratings (simulate user–activity interactions)
    # ==========================================================================
    severity_boost = {
        "Normal": 0.0, "Mild": 0.3, "Moderate": 0.6,
        "Severe": 1.0, "Extremely Severe": 1.4,
    }
    eff_mult = {"high": 1.3, "moderate": 1.0, "low": 0.7}

    category_list = ["Mindfulness", "Physical", "Creative", "Social", "Rest", "Cognitive", "Lifestyle"]

    ratings  = []
    base_date = datetime(2024, 1, 1)

    for user in users:
        uid   = user["user_id"]
        dep_b = severity_boost[user["depression_severity"]]
        anx_b = severity_boost[user["anxiety_severity"]]
        str_b = severity_boost[user["stress_severity"]]

        # Evidence-based category preference weights
        cat_w    = np.ones(len(category_list))
        cat_w[0] += (anx_b + str_b) * 0.4   # Mindfulness: anxiety + stress
        cat_w[1] += dep_b * 0.6              # Physical:    depression
        cat_w[2] += anx_b * 0.3              # Creative:    anxiety
        cat_w[3] += dep_b * 0.7              # Social:      depression
        cat_w[4] += str_b * 0.5              # Rest:        stress
        cat_w[5] += (dep_b + anx_b) * 0.3   # Cognitive:   depression + anxiety
        cat_w[6] += (dep_b + anx_b + str_b) * 0.1  # Lifestyle: all
        cat_w   *= np.random.uniform(0.7, 1.3, len(category_list))
        cat_w    = np.maximum(cat_w, 0.1)
        cat_w   /= cat_w.sum()

        # 8–15 ratings per user → total ~11,500 interactions (well above 5,000)
        n_ratings = random.randint(8, 15)
        rated = set()
        for _ in range(n_ratings * 6):
            if len(rated) >= n_ratings:
                break
            cat   = np.random.choice(category_list, p=cat_w)
            picks = [a["product_id"] for a in activities if a["category"] == cat]
            if picks:
                rated.add(random.choice(picks))

        for aid in rated:
            act    = next(a for a in activities if a["product_id"] == aid)
            target = act["stress_target"]
            em     = eff_mult[act["effectiveness"]]

            if target == "depression":
                boost = dep_b * 0.4 * em
            elif target == "anxiety":
                boost = anx_b * 0.4 * em
            elif target == "stress":
                boost = str_b * 0.4 * em
            else:
                boost = max(dep_b, anx_b, str_b) * 0.2 * em

            raw    = 3.0 + boost + np.random.normal(0, 0.7)
            rating = round(max(1.0, min(5.0, round(raw * 2) / 2)), 1)
            days   = random.randint(0, 364)
            ts     = (base_date + timedelta(days=days)).strftime("%Y-%m-%d")
            ratings.append({"user_id": uid, "product_id": aid, "rating": rating, "timestamp": ts})

    ratings_df = pd.DataFrame(ratings)
    print(f"  STEP 2 complete – {len(ratings_df)} user–activity interactions generated.")

    # ==========================================================================
    # STEP 3 — Save combined dataset
    # ==========================================================================
    products_df.to_csv(os.path.join(data_dir, "products.csv"),  index=False)
    users_df.to_csv(os.path.join(data_dir, "users.csv"),        index=False)
    ratings_df.to_csv(os.path.join(data_dir, "ratings.csv"),    index=False)
    print(f"  STEP 3 complete – saved to {data_dir}")
    print(f"  Summary: {len(products_df)} activities | {len(users_df)} users | {len(ratings_df)} ratings")
    return products_df, users_df, ratings_df


if __name__ == "__main__":
    generate_data()
