import openai
from openai import OpenAI
import streamlit as st
import json
import pandas as pd
import io
import re
import requests
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="Google Ads – Kampagne Generator", layout="wide")

# --- Robust HTTP GET with retries and longer timeout ---
@st.cache_data(ttl=3600)
def safe_get(url, retries=3, delay=2, timeout=15):
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e


from urllib.parse import urljoin, urlparse
import hashlib

@st.cache_data(ttl=3600)
def fetch_keyword_metrics(all_keywords, customer_id: str, yaml_path: str = "google-ads.yaml") -> dict:
    """
    Returnerer dict: { keyword_lower: {"monthly": int, "competition": str, "cpc_low": float, "cpc_high": float} }
    Bruger KeywordPlanIdeaService.GenerateKeywordIdeas.
    Google Ads API tillader max 20 keywords pr. request.
    """
    # Google Ads imports flyttet ind i funktionen
    from google.ads.googleads.client import GoogleAdsClient
    from google.ads.googleads.errors import GoogleAdsException
    result = {}
    keywords = [k.strip() for k in (all_keywords or []) if isinstance(k, str) and k.strip()]
    if not keywords:
        return result
    # --- Fjern støjord fra søgeord før Google Ads API ---
    noise_words = {
        "bestil", "klik", "og", "hent", "online", "åbningstider", "pris", "gave", "gavekort", "gavepakke", "julegave"
    }
    cleaned_keywords = []
    for k in keywords:
        words = [w for w in k.split() if w.lower() not in noise_words]
        if len(words) >= 1:
            cleaned_keywords.append(" ".join(words))
    keywords = cleaned_keywords
    # Vis debug-tabel over de søgeord der faktisk sendes til Google
    st.write("📤 Sender følgende søgeord til Google Keyword Planner:", keywords[:50])
    if not keywords:
        return result
    try:
        client = GoogleAdsClient.load_from_storage(yaml_path)
        service = client.get_service("KeywordPlanIdeaService")
        request_template = client.get_type("GenerateKeywordIdeasRequest")
        request_template.customer_id = str(customer_id)
        request_template.language = "languageConstants/1000"  # Dansk
        request_template.geo_target_constants.append("geoTargetConstants/2756")  # Danmark
        request_template.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH

        # Max 20 keywords pr. request
        CHUNK = 20
        import time
        total_batches = (len(keywords) + CHUNK - 1) // CHUNK
        progress = st.empty()
        for i in range(0, len(keywords), CHUNK):
            chunk = keywords[i:i + CHUNK]
            progress.text(f"Henter batch {i//CHUNK + 1}/{total_batches}…")
            req = client.get_type("GenerateKeywordIdeasRequest")
            req.customer_id = request_template.customer_id
            req.language = request_template.language
            req.keyword_plan_network = request_template.keyword_plan_network
            for g in request_template.geo_target_constants:
                req.geo_target_constants.append(g)
            req.keyword_seed.keywords.extend(chunk)

            try:
                response = service.generate_keyword_ideas(request=req)
                for idea in response:
                    text = idea.text
                    metrics = idea.keyword_idea_metrics
                    if not text:
                        continue
                    key = text.strip().lower()
                    monthly = int(metrics.avg_monthly_searches or 0)
                    # Robust håndtering af competition enum
                    try:
                        comp_enum = metrics.competition
                        competition = client.enums.KeywordPlanCompetitionLevelEnum.Name(comp_enum)
                    except Exception:
                        competition = "UNKNOWN"
                    low = (metrics.low_top_of_page_bid_micros or 0) / 1_000_000
                    high = (metrics.high_top_of_page_bid_micros or 0) / 1_000_000
                    result[key] = {
                        "monthly": monthly,
                        "competition": competition.title() if isinstance(competition, str) else str(competition),
                        "cpc_low": round(low, 2),
                        "cpc_high": round(high, 2),
                    }
            except GoogleAdsException as ex:
                st.warning(f"Google Ads batch fejlede: {ex}")
            except Exception as inner_e:
                st.warning(f"Fejl under batch {i//CHUNK + 1}: {inner_e}")

            # Lille pause mellem kald for sikkerhed
            time.sleep(0.5)

        progress.empty()
        return result

    except FileNotFoundError:
        st.warning("😕 Fandt ikke `google-ads.yaml` i projektmappen. Opret den først.")
    except GoogleAdsException as ex:
        st.error(f"Google Ads API fejl: {ex}")
    except Exception as e:
        st.error(f"Kunne ikke hente søgevolumen: {e}")
    return result

@st.cache_data(ttl=3600)
def fetch_semrush_metrics(keywords, api_key=None, database="dk"):
    # Bruger Generaxion Keyword API endpoint
    result = {}
    if not keywords:
        return result
    url = "https://niclasaccess.generaxion.dev/api/seo-analysis/batch-keyword-analysis"
    headers = {
        "Authorization": "Bearer token/Vami3KQpV0DUR1S18K4FomlfGgITrCFU/api/seo-analysis/batch-keyword-analysis",
        "Content-Type": "application/json"
    }
    try:
        payload = {"keywords": keywords}
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        if r.status_code != 200:
            st.error(f"Generaxion Keyword API fejl: {r.status_code}")
            st.code(r.text)
            st.stop()
        data = r.json()
        for item in data:
            kw = item.get("Keyword", "").lower()
            if not kw:
                continue
            result[kw] = {
                "monthly": item.get("Search Volume", 0),
                "cpc_dkk": round(item.get("CPC", 0.0), 2),
                "competition": round(item.get("Competition", 0.0), 2),
                "difficulty": 0.0,  # placeholder for konsistens
                "trends": item.get("Trends", []),
            }
        return result
    except Exception as e:
        st.error(f"Fejl ved Generaxion Keyword API: {e}")
        return {}


# Google Ads imports (flyttet ind i fetch_keyword_metrics)

# --- Stable hash of inputs to detect changes between analyse og generering ---
def compute_input_hash(xpect_text, customer_website, additional_info, geo_areas, campaign_types, budget) -> str:
    payload = json.dumps({
        "x": xpect_text or "",
        "w": customer_website or "",
        "a": additional_info or "",
        "g": geo_areas or "",
        "t": campaign_types or [],
        "b": budget,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

# --- Simple domain extraction (no external deps) ---
DOMAIN_REGEX = r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"

def extract_domains(text: str) -> list:
    """Extract plausible root domains (e.g., 'invita.dk') from arbitrary text.
    Lightweight, no external deps. Strips scheme/www and returns last two labels."""
    if not text:
        return []
    candidates = set()
    for token in re.findall(DOMAIN_REGEX, text or ""):
        token = token.lower().strip()
        token = token.replace("http://", "").replace("https://", "")
        # hostname only
        token = token.split("/")[0]
        token = token.lstrip(".")
        token = token.replace("www.", "")
        labels = [p for p in token.split(".") if p]
        if len(labels) >= 2:
            root = ".".join(labels[-2:])
            candidates.add(root)
    return list(candidates)

def get_external_domains_from_homepage(base_url: str, max_domains: int = 5) -> list:
    """Fetch homepage and collect external link root domains as a fallback."""
    try:
        resp = requests.get(base_url, timeout=6)
        s = BeautifulSoup(resp.text, "html.parser")
        base_domain = urlparse(base_url).netloc.replace("www.", "").lower()
        outs = set()
        for a in s.find_all("a", href=True):
            href = a["href"]
            full = urljoin(base_url, href)
            host = urlparse(full).netloc.lower()
            host = host.replace("www.", "")
            if host and base_domain not in host and host != base_domain:
                outs.update(extract_domains(full))
        # prune
        outs = [d for d in outs if d and d != base_domain]
        return sorted(outs)[:max_domains]
    except Exception:
        return []

st.title("Google Ads – Kampagne Generator")

# --- Sidebar: Model- og keyword-kildevalg (ALTID synlig) ---
with st.sidebar:
    model_choice = st.selectbox("Vælg GPT-model", ["gpt-5", "gpt-4o"], key="sidebar_model_choice")
    keyword_source = st.selectbox(
        "Vælg keyword-kilde",
        ["SEMrush", "Google Keyword Planner"],
        key="sidebar_keyword_source"
    )
    api_key = st.text_input("Indtast din OpenAI API-nøgle", type="password", key="sidebar_openai_key")
    # SEMrush bruger nu Generaxion Keyword API (ingen nøgle nødvendig)
    if model_choice and keyword_source:
        # Google Ads Customer ID (always visible)
        gads_customer_id = st.text_input(
            "Google Ads customer ID (uden bindestreger)",
            value="7445232535",
            key="sidebar_gads_customer_id"
        )
        semrush_key = ""  # SEMrush bruger nu Generaxion API, ikke individuel nøgle
    else:
        semrush_key = ""
        gads_customer_id = ""
    # Expose chosen data source name for later logic
    data_source = keyword_source


# --- Trinvist procesflow via st.session_state["step"] ---
if "step" not in st.session_state:
    st.session_state["step"] = "input"

# --- Input- og session-state felter ---
if "xpect_text" not in st.session_state:
    st.session_state["xpect_text"] = ""
if "customer_website" not in st.session_state:
    st.session_state["customer_website"] = ""
if "additional_info" not in st.session_state:
    st.session_state["additional_info"] = ""
if "geo_areas" not in st.session_state:
    st.session_state["geo_areas"] = ""
if "scraped_info" not in st.session_state:
    st.session_state["scraped_info"] = ""
if "total_daily_budget" not in st.session_state:
    st.session_state["total_daily_budget"] = 500
if "selected_campaign_types" not in st.session_state:
    st.session_state["selected_campaign_types"] = ["Search"]

#
# --- Fase 1: INPUT ---
if st.session_state["step"] == "input":
    st.header("1. Indsæt Xpect")
    uploaded_xpect = st.file_uploader("Upload Xpect (PDF, TXT eller DOCX)", type=["pdf", "txt", "docx"])
    if uploaded_xpect is not None:
        if uploaded_xpect.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_xpect)
            xpect_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif uploaded_xpect.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            from docx import Document
            doc = Document(uploaded_xpect)
            xpect_text = "\n".join([p.text for p in doc.paragraphs])
        else:
            xpect_text = uploaded_xpect.read().decode("utf-8", errors="ignore")
        st.success("✅ Xpect-fil uploadet")
    else:
        xpect_text = st.text_area("Kopier hele Xpect-teksten her", value=st.session_state.get("xpect_text", ""), key="xpect_input")
    st.session_state["xpect_text"] = xpect_text

    # Geografisk områdeudtræk fra Xpect
    def extract_cities_from_xpect(xpect_text):
        city_list = [
            "København","Aarhus","Århus","Odense","Aalborg","Esbjerg","Randers","Kolding","Horsens","Vejle","Roskilde",
            "Herning","Hørsholm","Silkeborg","Næstved","Fredericia","Viborg","Køge","Holstebro","Taastrup","Slagelse",
            "Hadsund","Hobro","Skanderborg","Grenaa","Ebeltoft","Ikast","Svendborg","Ringkøbing","Hillerød","Sønderborg",
            "Helsingør","Kalundborg","Thisted","Frederikshavn","Nykøbing","Holbæk","Nakskov","Middelfart","Rønne","Aabenraa"
        ]
        found = []
        text = xpect_text or ""
        for c in city_list:
            if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE):
                norm = "Aarhus" if c in ("Aarhus", "Århus") else c
                found.append(norm)
        return sorted(set([f.title() for f in found]))

    geo_areas_default = ""
    placeholder_text = ""
    if xpect_text:
        cities = extract_cities_from_xpect(xpect_text)
        if cities:
            geo_areas_default = ", ".join(cities)
        else:
            placeholder_text = "Ingen geografier fundet i Xpect"
            st.info("ℹ️ Ingen geografier fundet i Xpect. Du kan skrive dem manuelt (fx 'Randers, Hobro').")
    geo_areas = st.text_input(
        "Geografiske områder (kan redigeres)",
        value=geo_areas_default or st.session_state.get("geo_areas", ""),
        placeholder=placeholder_text,
        key="geo_input"
    )
    st.session_state["geo_areas"] = geo_areas

    st.header("2. Kundens website")
    customer_website = st.text_input("Indtast URL til kundens website", value=st.session_state.get("customer_website", ""), key="website_input")
    st.session_state["customer_website"] = customer_website

    # Website scraping (vis kun udtræk, ingen automatisk keyword scraping)
    scraped_info = ""
    RELEVANT_TAGS = {"h1", "h2", "h3", "title"}
    def get_internal_links(base_url, soup):
        base_domain = urlparse(base_url).netloc
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == base_domain:
                links.add(full_url)
        return list(links)
    def clean_scraped_texts(texts, base_domain=None):
        cleaned = []
        seen = set()
        noise_pattern = re.compile(
            r"(?i)\b(nyhed|klik her|læs mere|kontakt|book|campaya|snak|faq|bestil|webshop|ring|tilmelding)\b"
        )
        for t in texts:
            t_clean = re.sub(r'\s+', ' ', t.strip())
            if len(t_clean) < 15 or len(t_clean) > 120:
                continue
            if noise_pattern.search(t_clean):
                continue
            if base_domain:
                base = base_domain.split('.')[0]
                if base in t_clean.lower():
                    continue
            if "|" in t_clean:
                t_clean = t_clean.split("|")[0].strip()
            if t_clean.lower() in seen:
                continue
            seen.add(t_clean.lower())
            cleaned.append(t_clean)
        return cleaned
    if customer_website:
        try:
            response = safe_get(customer_website)
            soup = BeautifulSoup(response.text, "html.parser")
            all_links = get_internal_links(customer_website, soup)
            pages_to_scrape = [customer_website] + all_links[:4]
            all_texts = []
            start = time.time()

            def scrape_page(url):
                try:
                    res = safe_get(url)
                    sub_soup = BeautifulSoup(res.text, "html.parser")
                    return [h.get_text(strip=True) for h in sub_soup.find_all(RELEVANT_TAGS)]
                except Exception:
                    return []

            with ThreadPoolExecutor(max_workers=5) as ex:
                results = list(ex.map(scrape_page, pages_to_scrape))
                for headings in results:
                    all_texts.extend(headings)
                    if time.time() - start > 8:
                        st.warning("⏱️ Website-analyse stoppet efter 8 sekunder.")
                        break
            filtered = clean_scraped_texts(all_texts, base_domain=urlparse(customer_website).netloc)
            scraped_info = "\n".join(filtered[:50])
            st.write("🔎 Website analyseret – fundet følgende relevante indhold:")
            st.code(scraped_info)
        except Exception as e:
            st.warning(f"Kunne ikke analysere website: {e}")
    st.session_state["scraped_info"] = scraped_info

    st.header("3. Samlet dagsbudget")
    total_daily_budget = st.number_input(
        "Indtast samlet dagsbudget for alle kampagner (DKK)",
        min_value=1,
        value=st.session_state.get("total_daily_budget", 500),
        key="daily_budget_input_main"
    )
    st.session_state["total_daily_budget"] = total_daily_budget

    st.header("4. Ønskede kampagnetyper")
    selected_campaign_types = st.multiselect(
        "Vælg kampagnetyper du ønsker at inkludere",
        ["Search", "Display", "Performance Max", "Shopping", "Video"],
        default=st.session_state.get("selected_campaign_types", ["Search"]),
        key="campaign_types_multiselect_main"
    )
    st.session_state["selected_campaign_types"] = selected_campaign_types

    st.header("5. Anden vigtig information")
    additional_info = st.text_area("Skriv evt. yderligere info, som ikke står i Xpect", value=st.session_state.get("additional_info", ""), key="additional_info_input")
    st.session_state["additional_info"] = additional_info

    # --- Knap til at gå videre til analyse-fasen ---
    if st.button("🔍 Kør analyser"):
        # Gem input i session_state og gå til næste trin
        st.session_state["step"] = "analysis"
        st.rerun()

    # --- Session state defaults for analyses persistence ---
if "analysis_text" not in st.session_state:
    st.session_state["analysis_text"] = ""
if "competitor_analysis_text" not in st.session_state:
    st.session_state["competitor_analysis_text"] = ""
if "competitor_input" not in st.session_state:
    st.session_state["competitor_input"] = ""
if "analyses_ready" not in st.session_state:
    st.session_state["analyses_ready"] = False
if "analysis_hash" not in st.session_state:
    st.session_state["analysis_hash"] = ""

#
# --- Fase 2: ANALYSIS ---
if st.session_state["step"] == "analysis":
    st.header("Kør analyser")
    st.info("🤖 Analyserer input... dette kan tage et øjeblik.")
    xpect_text = st.session_state["xpect_text"]
    customer_website = st.session_state["customer_website"]
    additional_info = st.session_state["additional_info"]
    geo_areas = st.session_state["geo_areas"]
    selected_campaign_types = st.session_state["selected_campaign_types"]
    scraped_info = st.session_state["scraped_info"]
    total_daily_budget = st.session_state["total_daily_budget"]
    try:
        client = OpenAI(api_key=api_key)
        combined_analysis_prompt = f"""Du er en erfaren Google Ads strateg.
Baseret på nedenstående input (Xpect, website, website-indhold, ekstra noter, geografiske områder, ønskede kampagnetyper og budget), skal du udføre en samlet analyse bestående af:

1. **Foranalyse**:
   - Primære budskaber og value proposition
   - Målgruppe(r)
   - USP’er
   - Tone of voice
   - Call-to-actions
   - Potentielle annoncevinkler
   Skriv kort på dansk i punktform.

2. **Konkurrentforslag**:
   - Find de 2–5 vigtigste konkurrenter i Danmark for virksomheden ud fra Xpect og websitet.
   - Returnér KUN en kommasepareret liste over roddomæner (uden http/https og uden understier).

3. **Konkurrentanalyse**:
   - For hver nævnt konkurrent: Skriv kort på dansk (punktform) om deres primære budskaber, USP’er, CTA’er, tone og stil, samt eventuelle gaps.
   - Afslut med 3–5 forslag til hvordan virksomheden kan differentiere sig fra konkurrenterne.

**Outputformat**:
Returnér ét samlet JSON-objekt med følgende nøgler:
{{
  "foranalyse": "...",
  "konkurrenter": "domæne1.dk, domæne2.dk, ...",
  "konkurrentanalyse": "..."
}}

---

**Input:**
Xpect:
{xpect_text}

Website:
{customer_website}

Website-indhold:
{scraped_info}

Ekstra noter:
{additional_info}

Geografiske områder:
{geo_areas}

Ønskede kampagnetyper:
{", ".join(selected_campaign_types)}

Samlet dagsbudget:
{total_daily_budget} kr.
"""
        with st.spinner("🤖 Analyserer data…"):
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "Du er en erfaren Google Ads strateg, der laver foranalyse og konkurrentanalyse før kampagnestruktur."},
                    {"role": "user", "content": combined_analysis_prompt}
                ]
            )
        output = response.choices[0].message.content
        try:
            parsed = json.loads(output)
        except Exception:
            m = re.search(r"\{.*\}", output, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = {}
            else:
                parsed = {}
        foranalyse = parsed.get("foranalyse", "")
        konkurrenter = parsed.get("konkurrenter", "")
        konkurrentanalyse = parsed.get("konkurrentanalyse", "")
        suggested = extract_domains(konkurrenter)
        if not suggested:
            suggested = get_external_domains_from_homepage(customer_website, max_domains=5)
        own = urlparse(customer_website).netloc.replace("www.", "").lower()
        suggested = [d for d in suggested if d and d != own]
        st.session_state["competitor_input"] = ", ".join(sorted(set(suggested)))
        st.session_state["analysis_text"] = foranalyse
        st.session_state["competitor_analysis_text"] = konkurrentanalyse
        st.session_state["analyses_ready"] = True
        st.session_state["analysis_hash"] = compute_input_hash(
            xpect_text,
            customer_website,
            additional_info,
            geo_areas,
            selected_campaign_types,
            total_daily_budget,
        )
        st.subheader("📊 Foranalyse")
        st.write(foranalyse)
        st.subheader("🔎 Foreslåede konkurrenter")
        st.write(st.session_state["competitor_input"])
        st.subheader("🏁 Konkurrentanalyse")
        st.write(konkurrentanalyse)
        # Tilføj knap til at fortsætte til søgeordsudvælgelse
        if st.button("➡️ Fortsæt til søgeordsudvælgelse"):
            st.session_state["step"] = "keywords"
            st.rerun()
    except Exception as e:
        st.error(f"Analyse mislykkedes: {e}")

# --- Fase 3: KEYWORDS ---
if st.session_state["step"] == "keywords":
    st.header("Søgeord – udvælg og redigér")
    # Udtræk fra input
    xpect_text = st.session_state["xpect_text"]
    customer_website = st.session_state["customer_website"]
    additional_info = st.session_state["additional_info"]
    geo_areas = st.session_state["geo_areas"]
    selected_campaign_types = st.session_state["selected_campaign_types"]
    scraped_info = st.session_state["scraped_info"]
    total_daily_budget = st.session_state["total_daily_budget"]

    # --- GPT-baseret keyword forslag (forbedret prompt med niveauer, opdateret outputformat uden søgevolumen) ---
    keyword_prompt = f"""
Du er en erfaren dansk Google Ads-specialist. Udarbejd en komplet liste over **40–60 danske søgeord** til Google Search, opdelt i fire niveauer:

1. **Bredde søgeord** (generelle, overordnede termer for branchen/produktet)
2. **Produkt- og tjenesteniveau** (konkrete produkter, varianter, services, mærker)
3. **Købsintention** (søgeord hvor brugeren har tydelig købs- eller kontaktintention, fx 'køb', 'pris', 'bestil', 'tilbud', 'butik', 'leverandør', 'engros', 'b2b', 'grossist')
4. **Lokale varianter** (samme søgeord med tilføjede geografi/område – fx bynavne, regioner, landsdele)

**Outputformat:**
Returnér kun en ren liste, hvor hver linje har formatet:
<niveau>; <søgeord>

Eksempel:
Bred; spegepølser
Produkt; stikkelsbærsaft
Købsintention; køb gavekurv
Lokal; delikatesser aalborg

Brug kun **relevante og realistiske** søgeord, som danske brugere faktisk ville søge efter. Undgå irrelevante long-tails eller tekniske vendinger, og varier bredde og detaljeringsgrad.

**Ingen punktopstilling, ingen forklaringer, kun listen som beskrevet.**

Analyse:
{st.session_state['analysis_text']}

Konkurrentanalyse:
{st.session_state['competitor_analysis_text']}

Ekstra noter:
{st.session_state['additional_info']}
"""

    if "keywords_generated" in st.session_state:
        keywords_raw = st.session_state["keywords_generated"]
    else:
        with st.spinner("🤖 Genererer søgeordsforslag…"):
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "Du er en Google Ads-specialist, der laver keyword research på dansk."},
                    {"role": "user", "content": keyword_prompt}
                ]
            )
            keywords_raw = response.choices[0].message.content
            if not keywords_raw or len(keywords_raw.strip()) < 10:
                st.warning("AI returnerede ingen søgeord — ingen søgeord tilgængelige.")
                keywords_raw = ""
            st.session_state["keywords_generated"] = keywords_raw

    # Rens og normalisér listen
    gpt_keywords = [k.strip().lower() for k in keywords_raw.split("\n") if len(k.strip()) > 2]
    gpt_keywords = [k for k in gpt_keywords if not re.search(r"[^a-zæøå0-9\s\-\.;]", k)]
    gpt_keywords = [k for k in gpt_keywords if not any(word in k for word in ["analyse", "tone", "call-to-action", "primære", "usp", "målgruppe"])]

    # Udtræk kun søgeordet fra evt. niveau/volumen-format
    def extract_kw(line):
        # Forventet format: "<niveau>; <søgeord>; <volumen>"
        parts = [p.strip() for p in line.split(";")]
        if len(parts) >= 2:
            return parts[1]
        return line
    gpt_keywords = [extract_kw(k) for k in gpt_keywords if extract_kw(k)]
    # Fjern for lange/fragmenterede
    gpt_keywords = [k for k in gpt_keywords if 2 < len(k) <= 40 and len(k.split()) <= 5 and not k.startswith('-')]
    potential_keywords = sorted(set(gpt_keywords))

    st.subheader("📋 Foreslåede søgeord (før filtrering)")
    st.dataframe(pd.DataFrame({"Søgeord": potential_keywords}), use_container_width=True)

    with st.spinner("🔍 Henter søgeord..."):
        metrics_map = {}
        if potential_keywords:
            if data_source == "Google Keyword Planner" and gads_customer_id:
                metrics_map = fetch_keyword_metrics(potential_keywords, gads_customer_id)
            elif data_source == "SEMrush":
                with st.spinner("🔹 Henter søgedata fra Generaxion Keyword API…"):
                    metrics_map = fetch_semrush_metrics(potential_keywords, database="dk")
        def get_fallback_keywords(potential_keywords, metrics_map, min_count=10):
            valid = [k for k in potential_keywords if metrics_map.get(k.lower(), {}).get("monthly", 0) > 0]
            if len(valid) >= min_count:
                return valid
            extras = [k for k in potential_keywords if k not in valid]
            return valid + extras[:max(0, min_count - len(valid))]
        valid_keywords = get_fallback_keywords(potential_keywords, metrics_map, min_count=10)
        table_rows = []
        for k in valid_keywords:
            m = metrics_map.get(k.lower(), {})
            if data_source == "Google Keyword Planner":
                monthly = m.get("monthly", 0)
                cpc = m.get("cpc_low", 0.0)
                comp = m.get("competition", "")
                source = "Google"
            else:
                monthly = m.get("monthly", 0)
                cpc = m.get("cpc_dkk", 0.0)
                comp = m.get("competition", "")
                source = "SEMrush"
            table_rows.append({
                "Søgeord": k,
                "Månedlige søgninger": monthly,
                "CPC (DKK)": cpc,
                "Konkurrence": comp,
                "Datakilde": source,
            })
        df_keywords = pd.DataFrame(table_rows)
        df_valid = df_keywords[df_keywords["Månedlige søgninger"] > 0]
        st.subheader("🔑 Søgeord med søgevolumen")
        st.dataframe(df_valid, use_container_width=True)
        with st.expander("⚠️ Vis søgeord uden søgevolumen"):
            st.dataframe(df_keywords[df_keywords["Månedlige søgninger"] == 0], use_container_width=True)
        # Multiselect til valg/fravalg af søgeord
        approved = st.multiselect(
            "Vælg de søgeord du ønsker at inkludere",
            options=df_valid["Søgeord"].tolist(),
            default=df_valid["Søgeord"].tolist(),
            key="approved_keywords_multiselect"
        )
        st.session_state["approved_keywords"] = approved
        # Tekstfelt og knap til manuel tilføjelse af søgeord
        new_kw = st.text_area("Tilføj manuelt ekstra søgeord (ét pr. linje eller kommasepareret)", key="new_keywords_input")
        if st.button("➕ Tilføj søgeord"):
            extra = []
            for s in re.split(r"[,;\n]", new_kw):
                s = s.strip()
                if s and s not in st.session_state["approved_keywords"]:
                    extra.append(s)
            if extra:
                st.session_state["approved_keywords"].extend([k for k in extra if k and k not in st.session_state["approved_keywords"]])
                st.session_state["approved_keywords"] = sorted(set(st.session_state["approved_keywords"]))
                st.success(f"Tilføjede {len(extra)} nye søgeord.")
            else:
                st.info("Ingen nye unikke søgeord fundet.")
        # Mulighed for at fjerne søgeord
        if st.button("🔁 Opdater søgevolumen for valgte søgeord"):
            keywords_to_update = st.session_state["approved_keywords"]
            if not keywords_to_update:
                st.warning("Ingen valgte søgeord.")
            else:
                with st.spinner("🔍 Henter søgeord..."):
                    if data_source == "Google Keyword Planner" and gads_customer_id:
                        updated_metrics = fetch_keyword_metrics(keywords_to_update, gads_customer_id)
                        metrics_map.update(updated_metrics)
                    elif data_source == "SEMrush":
                        if semrush_key:
                            updated_metrics = fetch_semrush_metrics(keywords_to_update, semrush_key, database="dk")
                            metrics_map.update(updated_metrics)
                        else:
                            st.info("🔹 SEMrush-data hentes.")
                st.success("Søgevolumen opdateret for valgte søgeord.")
        # Gem metrics_map i session_state til næste fase
        st.session_state["metrics_map"] = metrics_map
        # Knap til at godkende søgeord og gå videre
        if st.button("✅ Godkend søgeord"):
            st.session_state["step"] = "generation"
            st.experimental_set_query_params(step="generation")
            st.rerun()


def extract_json_from_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    patterns = [r"\{.*\"campaigns\".*\}", r"\{.*\}", r"\[.*\]"]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    raise ValueError("Kunne ikke finde gyldigt JSON i AI-outputtet.")

# --- Utility: Stringify a list for output in dataframe ---
def stringify_list(items):
    if not items:
        return ""
    result = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text") or item.get("title") or str(item)
            result.append(text)
        else:
            result.append(str(item))
    return "; ".join(result)


def normalize_ad_obj(ad: dict, customer_website: str) -> dict:
    """
    Normalize ad payloads from the model into a consistent shape for downstream use.
    Accepts an ad dictionary and ensures keys like headline_1..9, description_1..4, final_url, path_1, path_2 are present.
    """
    if not isinstance(ad, dict):
        return {}
    out = dict(ad)  # shallow copy

    # Headlines may come as a list under 'headlines'
    headlines = []
    if isinstance(out.get("headlines"), list):
        headlines = out.get("headlines")
    else:
        # try to collect headline_1..headline_15 if present
        for i in range(1, 16):
            key = f"headline_{i}"
            if key in out and isinstance(out[key], str):
                headlines.append(out[key])
    for i in range(9):
        out[f"headline_{i+1}"] = (headlines[i] if i < len(headlines) else out.get(f"headline_{i+1}", ""))

    # Descriptions may come as a list under 'descriptions'
    descriptions = []
    if isinstance(out.get("descriptions"), list):
        descriptions = out.get("descriptions")
    else:
        for i in range(1, 9):
            key = f"description_{i}"
            if key in out and isinstance(out[key], str):
                descriptions.append(out[key])
    for i in range(4):
        out[f"description_{i+1}"] = (descriptions[i] if i < len(descriptions) else out.get(f"description_{i+1}", ""))

    # Final URL can be under multiple keys
    final_url = (
        out.get("final_url")
        or out.get("finalUrl")
        or (out.get("finalUrls")[0] if isinstance(out.get("finalUrls"), list) and out.get("finalUrls") else None)
        or out.get("url")
        or out.get("landing_page")
        or customer_website
    )
    out["final_url"] = final_url or customer_website or ""

    # Paths may vary
    out["path_1"] = out.get("path_1") or out.get("path1") or out.get("path") or ""
    out["path_2"] = out.get("path_2") or out.get("path2") or ""

    return out

# --- Fase 4: GENERATION ---
if st.session_state["step"] == "generation":
    st.header("Generér kampagnestruktur")
    st.info("Kampagnestrukturen bygges på baggrund af de valgte søgeord og analyser.")
    if st.button("💾 Gem og fortsæt"):
        # Kør kampagnestrukturbygning
        # (Flyttet fra tidligere "Gem og fortsæt"-knap)
        xpect_text = st.session_state["xpect_text"]
        customer_website = st.session_state["customer_website"]
        additional_info = st.session_state["additional_info"]
        geo_areas = st.session_state["geo_areas"]
        selected_campaign_types = st.session_state["selected_campaign_types"]
        scraped_info = st.session_state["scraped_info"]
        total_daily_budget = st.session_state["total_daily_budget"]
        analysis_for_prompt = st.session_state.get("analysis_text", "")
        competitor_analysis_for_prompt = st.session_state.get("competitor_analysis_text", "")
        approved_keywords = st.session_state.get("approved_keywords", [])
        # Fallback: hent metrics_map fra session_state hvis muligt
        metrics_map = st.session_state.get("metrics_map", {})
        # Ny filtreringslogik for approved_keywords ifm. metrics_map
        approved_keywords = approved_keywords or []
        if metrics_map:
            valid_kw = [kw for kw in approved_keywords if metrics_map.get(kw.lower(), {}).get("monthly", 0) > 0]
            if not valid_kw:
                st.warning("Ingen søgeord med søgevolumen > 0 fundet – fortsætter med alle godkendte søgeord.")
                valid_kw = approved_keywords
            approved_keywords = valid_kw
        else:
            st.warning("Ingen søgedata fundet – fortsætter med alle godkendte søgeord.")
        # Visuel bekræftelse på antal søgeord inkluderet
        st.info(f"📊 {len(approved_keywords)} søgeord inkluderet i kampagnestrukturen.")
        system_prompt = """\
Create a detailed Google Ads campaign structure for a new client using the provided requirements specification (Xpect), URL, analysis, and extra notes as input. Your tasks are to:

- Propose the following components:
  - **Campaign structure:** Specify type(s) (Search/Display), and note if it's brand vs. generic.
  - **Campaign names**
  - **Ad groups** (with clear theme and focus)
  - **Keywords:** Up to 10 for each ad group
  - **Daily budget** for each campaign
  - **One ad per ad group** with at least 9 unique, high-quality Danish headlines and 4 descriptions.
  - **Campaign extensions:** Always include exactly 4 sitelinks, 3-4 callout extensions, structured snippets, and call extensions.

#### Important ad copy guidelines (headlines & descriptions):

- Write short, complete sentences (20–30 characters per headline).
- Do not truncate in the middle of words.
- All headlines must be full, meaningful sentences for users.
- **Never end** a headline or description with a single Danish function word: “i”, “på”, “til”, “for”, “med”, “om”, “af”, “hos”, “ved”, “uden”, “over”, “under”, “mellem”.
- Use natural, flowing, and professional Danish as seen in real Google Ads.
- Only capitalize at the beginning of sentences or for proper nouns.
- Descriptions (max 90 characters) must be full, meaningful sentences—never cut off or incomplete.
- Avoid repetition, filler, or partial sentences.

**Good headline examples (under 30 characters):**
- “Flyt nemt og sikkert”
- “Få gratis flyttetilbud”
- “Transport i hele Jylland”
- “Professionel flyttehjælp”

**Good description examples:**
- “Vi tilbyder tryg og hurtig flytning i hele Danmark.”
- “Få et gratis tilbud og flyt uden besvær.”
- “Erfaren flyttehjælp til private og erhverv.”

---

### Output Format

- Output MUST be in JSON format with the top key: **"campaigns"**
- Each campaign must include: campaign type, name, daily budget, ad groups (with name, theme, keywords), ad (with 9+ headlines, 4+ descriptions), and extensions (sitelinks, callouts, structured snippets, call).
- Do NOT add explanations or text outside the JSON structure.

---

### Reasoning and Output Ordering

1. **Reasoning and Drafting:** Internally, first analyze all input information (brief/specification, URL, analysis, extra notes), select relevant themes, and define a logical account and campaign structure. Then, for each element (ad groups, keywords, extensions, etc.), determine optimal content based on the input. Ensure all ad text rules are followed and content is audience-appropriate.
2. **Conclusion/Output:** Only after the complete structure and copy are finalized, output the full JSON as described.

**NB:** Your output should consist exclusively of the final “campaigns” JSON, meeting all detailed requirements.
"""
        user_prompt = f"""\
Xpect:
{xpect_text}

Website:
{customer_website}

Analyse (fra AI):
{analysis_for_prompt}

Konkurrentanalyse (fra AI):
{competitor_analysis_for_prompt}

Ekstra noter:
{additional_info}

Geografisk fokus:
{geo_areas}

Website-indhold fundet ved scanning:
{scraped_info}

Ønskede kampagnetyper: {", ".join(selected_campaign_types)}.
Samlet dagsbudget må ikke overstige {total_daily_budget} kr.
Lav kampagnestrukturen så der laves kampagner eller annoncegrupper for alle nævnte geografiske områder. 
Hvis budgettet er lavt, fordel det jævnt mellem områderne.
Brug kun følgende søgeord (med bekræftet søgevolumen): {', '.join(approved_keywords) if approved_keywords else '[Ingen validerede søgeord fundet]'}
"""
        try:
            if not api_key:
                st.error("Indtast din OpenAI API-nøgle i sidebaren for at køre AI-analysen.")
                raise RuntimeError("Missing API key")
            with st.spinner("🧠 Genererer kampagnestruktur…"):
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                output_text = response.choices[0].message.content
            try:
                data = extract_json_from_text(output_text)
            except ValueError as e:
                st.error(str(e))
                data = None
            if not data:
                st.info("AI genererede ingen gyldig kampagnestruktur.")
                st.stop()
            # --- Hent søgevolumen/CPC/konkurrence for alle keywords (gentag evt. for endelig struktur) ---
            all_keywords = []
            for c in data.get("campaigns", []):
                for ag in c.get("ad_groups", []) or []:
                    all_keywords.extend((ag.get("keywords") or []))
            unique_keywords = sorted(set([k.strip() for k in all_keywords if isinstance(k, str) and k.strip()]))
            if unique_keywords:
                missing_metrics = [k for k in unique_keywords if k.lower() not in metrics_map]
                if missing_metrics:
                    if data_source == "Google Keyword Planner" and gads_customer_id:
                        extra_metrics = fetch_keyword_metrics(missing_metrics, gads_customer_id)
                        metrics_map.update(extra_metrics)
                        st.session_state["metrics_map"] = metrics_map
                    elif data_source == "SEMrush":
                        if semrush_key:
                            extra_metrics = fetch_semrush_metrics(missing_metrics, semrush_key, database="dk")
                            metrics_map.update(extra_metrics)
                            st.session_state["metrics_map"] = metrics_map
            rows = []
            TRAILING_STOPWORDS = {
                "i","på","til","for","med","om","af","hos","ved","uden","over","under","mellem"
            }
            def soft_trim(text: str, max_len: int) -> str:
                if not text:
                    return ""
                t = text.strip()
                if len(t) <= max_len:
                    return t
                cut = t[:max_len]
                if " " in cut:
                    cut = cut.rsplit(" ", 1)[0]
                return cut.strip()
            def polish_line(text: str, max_len: int) -> str:
                if not text:
                    return ""
                t = soft_trim(text, max_len)
                t = t.strip(" -·•,;:")
                parts = t.split()
                if parts and parts[-1].lower() in TRAILING_STOPWORDS:
                    t = " ".join(parts[:-1]).strip()
                return t
            def clean_asset(text, m):
                if not text:
                    return ""
                return polish_line(text, m)
            for campaign in data.get("campaigns", []):
                campaign_name = campaign.get("name", "")
                campaign_type = campaign.get("type", "")
                budget = campaign.get("daily_budget", "")
                sitelinks = campaign.get("sitelinks", [])
                callouts = campaign.get("callouts", [])
                structured_snippets = campaign.get("structured_snippets", [])
                call_extensions = campaign.get("call_extensions", [])
                for ad_group in campaign.get("ad_groups", []):
                    ad_group_name = ad_group.get("name", "")
                    keywords = ad_group.get("keywords", []) or []
                    ads = ad_group.get("ads", []) or []
                    ad = ads[0] if ads else {}
                    if not ad and isinstance(ad_group.get("ad"), dict):
                        ad = ad_group.get("ad")
                    if not ad and isinstance(ad_group.get("rsa"), dict):
                        ad = ad_group.get("rsa")
                    ad = normalize_ad_obj(ad, customer_website)
                    ad_headlines = [clean_asset(ad.get(f"headline_{i}", ""), 30) for i in range(1, 10)]
                    ad_descriptions = [clean_asset(ad.get(f"description_{i}", ""), 90) for i in range(1,5)]
                    final_url = ad.get("final_url", customer_website or "")
                    pinned_headlines = ad.get("pinned_headlines", {}) if isinstance(ad.get("pinned_headlines"), dict) else {}
                    base_row = {
                        "Campaign": campaign_name,
                        "Campaign Type": campaign_type,
                        "Campaign Status": "Enabled",
                        "Budget": budget,
                        "Budget type": "Daily",
                        "Networks": "Google search",
                        "Languages": "Danish;English",
                        "Ad Group": ad_group_name,
                        "Ad Group Status": "Enabled",
                        "Ad type": "Responsive search ad",
                        "Criterion Type": "",
                        "Status": "Enabled",
                        "Keyword": "",
                        "Keyword Match Type": "",
                        "Keyword Status": "",
                        "Headline 1": ad_headlines[0],
                        "Headline 2": ad_headlines[1],
                        "Headline 3": ad_headlines[2],
                        "Headline 4": ad_headlines[3],
                        "Headline 5": ad_headlines[4],
                        "Headline 6": ad_headlines[5],
                        "Headline 7": ad_headlines[6],
                        "Headline 8": ad_headlines[7],
                        "Headline 9": ad_headlines[8],
                        "Headline 1 Pin": pinned_headlines.get("headline_1", ""),
                        "Headline 2 Pin": pinned_headlines.get("headline_2", ""),
                        "Headline 3 Pin": pinned_headlines.get("headline_3", ""),
                        "Headline 4 Pin": pinned_headlines.get("headline_4", ""),
                        "Headline 5 Pin": pinned_headlines.get("headline_5", ""),
                        "Headline 6 Pin": pinned_headlines.get("headline_6", ""),
                        "Headline 7 Pin": pinned_headlines.get("headline_7", ""),
                        "Headline 8 Pin": pinned_headlines.get("headline_8", ""),
                        "Headline 9 Pin": pinned_headlines.get("headline_9", ""),
                        "Description 1": ad_descriptions[0],
                        "Description 2": ad_descriptions[1],
                        "Description 3": ad_descriptions[2],
                        "Description 4": ad_descriptions[3],
                        "Description 1 Pin": ad.get("pinned_descriptions", {}).get("description_1", ""),
                        "Description 2 Pin": ad.get("pinned_descriptions", {}).get("description_2", ""),
                        "Description 3 Pin": ad.get("pinned_descriptions", {}).get("description_3", ""),
                        "Description 4 Pin": ad.get("pinned_descriptions", {}).get("description_4", ""),
                        "Final URL": final_url,
                        "Path 1": ad.get("path_1", ""),
                        "Path 2": ad.get("path_2", ""),
                        "Sitelinks": stringify_list(sitelinks),
                        "Callouts": stringify_list(callouts),
                        "Structured Snippets": stringify_list(structured_snippets),
                        "Call Extensions": stringify_list(call_extensions),
                        "Monthly Searches": "",
                        "Competition": "",
                        "Top of page bid (low)": "",
                        "Top of page bid (high)": "",
                    }
                    rows.append(base_row)
                    for keyword in keywords:
                        kw_row = base_row.copy()
                        kw_row.update({
                            "Ad type": "",
                            "Criterion Type": "Exact",
                            "Status": "Enabled",
                            "Keyword": keyword,
                            "Keyword Match Type": "Exact",
                            "Keyword Status": "Enabled",
                            "Headline 1": "",
                            "Headline 2": "",
                            "Headline 3": "",
                            "Headline 4": "",
                            "Headline 5": "",
                            "Headline 6": "",
                            "Headline 7": "",
                            "Headline 8": "",
                            "Headline 9": "",
                            "Headline 1 Pin": "",
                            "Headline 2 Pin": "",
                            "Headline 3 Pin": "",
                            "Headline 4 Pin": "",
                            "Headline 5 Pin": "",
                            "Headline 6 Pin": "",
                            "Headline 7 Pin": "",
                            "Headline 8 Pin": "",
                            "Headline 9 Pin": "",
                            "Description 1": "",
                            "Description 2": "",
                            "Description 3": "",
                            "Description 4": "",
                            "Description 1 Pin": "",
                            "Description 2 Pin": "",
                            "Description 3 Pin": "",
                            "Description 4 Pin": "",
                            "Final URL": "",
                            "Path 1": "",
                            "Path 2": "",
                            "Sitelinks": "",
                            "Callouts": "",
                            "Structured Snippets": "",
                            "Call Extensions": "",
                            "Monthly Searches": (metrics_map.get(keyword.lower(), {}) or {}).get("monthly", 0),
                            "Competition": (metrics_map.get(keyword.lower(), {}) or {}).get("competition", ""),
                            "Top of page bid (low)": (metrics_map.get(keyword.lower(), {}) or {}).get("cpc_low", 0.0),
                            "Top of page bid (high)": (metrics_map.get(keyword.lower(), {}) or {}).get("cpc_high", 0.0),
                        })
                        rows.append(kw_row)
            if rows:
                df_ads = pd.DataFrame(rows)
                csv_buffer = io.BytesIO()
                df_ads.to_csv(csv_buffer, index=False, sep="\t", encoding="utf-16", lineterminator="\n")
                csv_buffer.seek(0)
                st.success("✅ Kampagnestruktur genereret! Download filen nedenfor.")
                st.download_button(
                    label="Download CSV til Ads Editor",
                    data=csv_buffer,
                    file_name="ads_editor_upload.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Fejl under AI-kald: {e}")




# --- Normalize ad payloads from the model into a consistent shape ---

def headline_cleanup(texts, api_key, model_choice, max_len=30, label="overskrifter"):
    """Auto-korrigerer korte eller ufuldstændige overskrifter/beskrivelser."""
    if not texts:
        return texts
    bad = [t for t in texts if re.search(r'\b(i|på|til|for|med)$', t.lower()) or len(t) < 15]
    if not bad:
        return texts
    prompt = f"Omskriv disse {label} til hele, men korte danske sætninger. Maks {max_len} tegn, og sætningen skal give mening:\n" + "\n".join(bad)
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Du er en dansk tekstforfatter for Google Ads. Du skriver fængende, men korte sætninger."},
                {"role": "user", "content": prompt}
            ]
        )
        suggestions = [s.strip() for s in resp.choices[0].message.content.split("\n") if s.strip()]
        idx = 0
        for i, t in enumerate(texts):
            if t in bad and idx < len(suggestions):
                texts[i] = suggestions[idx][:max_len].strip()
                idx += 1
    except Exception as e:
        st.warning(f"Kunne ikke forbedre {label}: {e}")
    return texts

#



