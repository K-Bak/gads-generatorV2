import openai
import streamlit as st
import json
import pandas as pd
import io
import re
import requests
from bs4 import BeautifulSoup


from urllib.parse import urljoin, urlparse
import hashlib

# Google Ads imports
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

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

st.set_page_config(page_title="Google Ads – Kampagne Generator", layout="wide")
st.title("Google Ads – Kampagne Generator")

st.header("1. Indsæt Xpect")

xpect_text = st.text_area("Kopier hele Xpect-teksten her")

# --- Geografisk områdeudtræk fra Xpect ---
def extract_cities_from_xpect(xpect_text):
    """Find simple Danish city names in Xpect text."""
    city_list = [
        "København","Aarhus","Århus","Odense","Aalborg","Esbjerg","Randers","Kolding","Horsens","Vejle","Roskilde",
        "Herning","Hørsholm","Silkeborg","Næstved","Fredericia","Viborg","Køge","Holstebro","Taastrup","Slagelse",
        "Hadsund","Hobro","Skanderborg","Grenaa","Ebeltoft","Ikast","Svendborg","Ringkøbing","Hillerød","Sønderborg",
        "Helsingør","Kalundborg","Thisted","Frederikshavn","Nykøbing","Holbæk","Nakskov","Middelfart","Rønne","Aabenraa"
    ]
    found = []
    text = xpect_text or ""
    for c in city_list:
        # Correct word-boundary regex (use re.escape and single \b)
        if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE):
            # Normalize Århus/Aarhus to Aarhus, and title-case others
            norm = "Aarhus" if c in ("Aarhus", "Århus") else c
            found.append(norm)
    # Unique + title case output
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
    value=geo_areas_default,
    placeholder=placeholder_text
)

st.header("2. Kundens website")
customer_website = st.text_input("Indtast URL til kundens website")

def get_internal_links(base_url, soup):
    base_domain = urlparse(base_url).netloc
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc == base_domain:
            links.add(full_url)
    return list(links)

scraped_info = ""
if customer_website:
    try:
        response = requests.get(customer_website, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        all_links = get_internal_links(customer_website, soup)
        pages_to_scrape = [customer_website] + all_links[:4]  # i alt 5 sider inkl. forsiden

        all_headings_links = []

        for url in pages_to_scrape:
            try:
                res = requests.get(url, timeout=5)
                sub_soup = BeautifulSoup(res.text, "html.parser")
                headings = [h.get_text(strip=True) for h in sub_soup.find_all(["h1", "h2", "h3"])]
                links = [a.get_text(strip=True) for a in sub_soup.find_all("a") if a.get_text(strip=True)]
                all_headings_links.extend(headings[:5] + links[:5])
            except Exception:
                continue

        scraped_info = "\n".join(all_headings_links[:50])
        st.write("🔎 Website analyseret – fundet følgende indhold på flere sider:")
        st.code(scraped_info)
    except Exception as e:
        st.warning(f"Kunne ikke analysere website: {e}")

st.header("3. Samlet dagsbudget")
total_daily_budget = st.number_input("Indtast samlet dagsbudget for alle kampagner (DKK)", min_value=1, value=500)

st.header("4. Ønskede kampagnetyper")
selected_campaign_types = st.multiselect(
    "Vælg kampagnetyper du ønsker at inkludere",
    ["Search", "Display", "Performance Max", "Shopping", "Video"],
    default=["Search"]
)

st.header("5. Anden vigtig information")
additional_info = st.text_area("Skriv evt. yderligere info, som ikke står i Xpect")

# OpenAI API key input in sidebar
api_key = st.sidebar.text_input("Indtast din OpenAI API-nøgle", type="password")

# model selection



model_choice = st.sidebar.selectbox("Vælg model", options=["gpt-4o", "gpt-4o-mini"], index=0)

# Google Ads – Customer ID (bruges til Keyword Planner)
gads_customer_id = st.sidebar.text_input("Google Ads customer ID (uden bindestreger)", value="7445232535")

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

# --- NYT TRIN 6: ANALYSEPAKKE (Kør samlet) ---
st.header("6. Kør analyser (valgfrit men anbefalet)")
st.info("🔍 Indtast Xpect, website, budget, vigtig info og din API-nøgle før du kører analyserne.")
run_all_analyses = st.button("🚀 Kør analyser")
analysis_text = ""
competitor_analysis_text = ""
if run_all_analyses and api_key and customer_website and xpect_text:
    try:
        openai.api_key = api_key
        # Step 1: Foranalyse
        st.subheader("📊 Kører foranalyse ...")
        analysis_prompt = f"""Du er en marketingstrateg med speciale i Google Ads.
Baseret på nedenstående input (Xpect + websitet), skal du udlede:
- Primære budskaber og value proposition
- Målgruppe(r)
- USP’er
- Tone of voice
- Call-to-actions
- Potentielle annoncevinkler
Skriv kort på dansk i punktform.

Xpect:
{xpect_text}

Website-indhold:
{scraped_info}

Evt. noter:
{additional_info}"""
        analysis_response = openai.ChatCompletion.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "Du er en erfaren Google Ads strateg, der laver indledende analyse før kampagnestruktur."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=1200,
            temperature=0.6
        )
        analysis_text = analysis_response.choices[0].message["content"]
        st.write(analysis_text)

        # Step 2: Konkurrentforslag (auto-suggest)
        st.subheader("🧭 Finder konkurrenter ...")
        suggest_prompt = f"""Du er en marketingstrateg. Baseret på virksomheden her: {customer_website}
Nedenfor er uddrag fra websitet samt en kort Xpect. Find 2–5 vigtigste konkurrenter i Danmark.
Returnér KUN en kommasepareret liste over roddomæner (uden http/https og uden understier).

Website-uddrag:
{scraped_info}

Xpect (kort):
{xpect_text[:1200]}"""
        suggest_response = openai.ChatCompletion.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "Du laver kort, præcis konkurrentliste (kun domæner, komma-separeret)."},
                {"role": "user", "content": suggest_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        suggestion_text = suggest_response.choices[0].message["content"]
        suggested = extract_domains(suggestion_text)
        if not suggested:
            suggested = get_external_domains_from_homepage(customer_website, max_domains=5)
        own = urlparse(customer_website).netloc.replace("www.", "").lower()
        suggested = [d for d in suggested if d and d != own]
        st.session_state["competitor_input"] = ", ".join(sorted(set(suggested)))
        st.write("🔎 Foreslåede konkurrenter:", st.session_state["competitor_input"])

        # Step 3: Konkurrentanalyse
        st.subheader("🏁 Kører konkurrentanalyse ...")
        competitor_input = st.session_state.get("competitor_input", "")
        competitor_analysis_prompt = f"""Du er en ekspert i Google Ads konkurrentanalyse.
Baseret på disse inputs (konkurrent-domæner og/eller søgeord):
{competitor_input}

Lever en kort dansk analyse der for hver dominerende konkurrent angiver:
- Primære budskaber
- USP’er
- CTA’er
- Tone og stil
- Gaps
Afslut med 3–5 forslag til hvordan vi kan differentiere os."""
        ca_response = openai.ChatCompletion.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "Du er en erfaren Google Ads strateg, der laver konkurrentanalyse for at informere kampagneopbygning."},
                {"role": "user", "content": competitor_analysis_prompt}
            ],
            max_tokens=1200,
            temperature=0.6
        )
        competitor_analysis_text = ca_response.choices[0].message["content"]
        st.write(competitor_analysis_text)
        st.session_state["analysis_text"] = analysis_text
        st.session_state["competitor_analysis_text"] = competitor_analysis_text
        st.session_state["analyses_ready"] = True
        st.session_state["analysis_hash"] = compute_input_hash(
            xpect_text,
            customer_website,
            additional_info,
            geo_areas,
            selected_campaign_types,
            total_daily_budget,
        )
    except Exception as e:
        st.warning(f"Analyse mislykkedes: {e}")
else:
    if run_all_analyses:
        st.warning("⚠️ Udfyld Xpect, Website og API-nøgle før du kører analyserne.")

# --- FUNKTIONER SOM FØR ---
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


# --- Normalize ad payloads from the model into a consistent shape ---
def normalize_ad_obj(ad: dict, customer_website: str) -> dict:
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

def headline_cleanup(texts, api_key, model_choice, max_len=30, label="overskrifter"):
    """Auto-korrigerer korte eller ufuldstændige overskrifter/beskrivelser."""
    if not texts:
        return texts
    bad = [t for t in texts if re.search(r'\b(i|på|til|for|med)$', t.lower()) or len(t) < 15]
    if not bad:
        return texts
    prompt = f"Omskriv disse {label} til hele, men korte danske sætninger. Maks {max_len} tegn, og sætningen skal give mening:\n" + "\n".join(bad)
    try:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "Du er en dansk tekstforfatter for Google Ads. Du skriver fængende, men korte sætninger."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.6
        )
        suggestions = [s.strip() for s in resp.choices[0].message["content"].split("\n") if s.strip()]
        idx = 0
        for i, t in enumerate(texts):
            if t in bad and idx < len(suggestions):
                texts[i] = suggestions[idx][:max_len].strip()
                idx += 1
    except Exception as e:
        st.warning(f"Kunne ikke forbedre {label}: {e}")
    return texts

#
# --- Google Ads Keyword Planner: hent søgevolumen/CPC/konkurrence ---
def fetch_keyword_metrics(all_keywords, customer_id: str, yaml_path: str = "google-ads.yaml") -> dict:
    """Returnerer dict: { keyword_lower: {"monthly": int, "competition": str, "cpc_low": float, "cpc_high": float} }
    Bruger KeywordPlanIdeaService.GenerateKeywordIdeas.
    """
    result = {}
    keywords = [k.strip() for k in (all_keywords or []) if isinstance(k, str) and k.strip()]
    if not keywords:
        return result
    try:
        client = GoogleAdsClient.load_from_storage(yaml_path)
        service = client.get_service("KeywordPlanIdeaService")
        request = client.get_type("GenerateKeywordIdeasRequest")
        # Customer ID (uden bindestreger)
        request.customer_id = str(customer_id)
        # Dansk (1000) og Danmark (2756)
        request.language = "languageConstants/1000"
        request.geo_target_constants.append("geoTargetConstants/2756")
        request.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH

        # API'en tillader op til ca. 1000 keywords ad gangen – chunk for en sikkerheds skyld
        CHUNK = 800
        for i in range(0, len(keywords), CHUNK):
            chunk = keywords[i:i+CHUNK]
            req = client.get_type("GenerateKeywordIdeasRequest")
            req.customer_id = request.customer_id
            req.language = request.language
            req.keyword_plan_network = request.keyword_plan_network
            for g in request.geo_target_constants:
                req.geo_target_constants.append(g)
            req.keyword_seed.keywords.extend(chunk)
            response = service.generate_keyword_ideas(request=req)
            for idea in response:
                text = idea.text
                metrics = idea.keyword_idea_metrics
                if not text:
                    continue
                key = text.strip().lower()
                monthly = int(metrics.avg_monthly_searches or 0)
                comp_enum = metrics.competition  # Enum value
                competition = client.enums.KeywordPlanCompetitionLevelEnum.Name(comp_enum)
                # CPC kommer i micros (DKK * 1e6). Brug top-of-page BID low/high, hvis sat.
                low = (metrics.low_top_of_page_bid_micros or 0) / 1_000_000
                high = (metrics.high_top_of_page_bid_micros or 0) / 1_000_000
                result[key] = {
                    "monthly": monthly,
                    "competition": competition.title() if isinstance(competition, str) else str(competition),
                    "cpc_low": round(low, 2),
                    "cpc_high": round(high, 2),
                }
        return result
    except FileNotFoundError:
        st.warning("😕 Fandt ikke `google-ads.yaml` i projektmappen. Opret den først.")
    except GoogleAdsException as ex:
        st.error(f"Google Ads API fejl: {ex}")
    except Exception as e:
        st.error(f"Kunne ikke hente søgevolumen: {e}")
    return result

# --- KAMPAGNEGENERERING ---
if st.button("Gem og fortsæt"):
    # --- Validate analyses state and show helpful hints ---
    current_hash = compute_input_hash(
        xpect_text,
        customer_website,
        additional_info,
        geo_areas,
        selected_campaign_types,
        total_daily_budget,
    )
    if st.session_state.get("analysis_hash") and st.session_state["analysis_hash"] != current_hash:
        st.info("⚠️ Du har ændret input siden sidste analyse. Overvej at køre analyserne igen for bedst resultat.")
    if not st.session_state.get("analyses_ready"):
        st.warning("Tip: Kør analyser først for skarpere output (jeg kan godt fortsætte uden, men kvaliteten bliver bedre med analyserne).")
    with st.expander("🔧 Debug – status for analyser"):
        st.write({
            "analyses_ready": st.session_state.get("analyses_ready"),
            "analysis_len": len(st.session_state.get("analysis_text", "")),
            "competitor_analysis_len": len(st.session_state.get("competitor_analysis_text", "")),
            "analysis_hash": st.session_state.get("analysis_hash"),
            "current_hash": current_hash,
        })
    input_data = {
        "xpect": xpect_text,
        "website": customer_website,
        "extra": additional_info,
        "budget": total_daily_budget,
        "campaign_types": selected_campaign_types,
        "analysis": st.session_state.get("analysis_text", ""),
        "competitor_analysis": st.session_state.get("competitor_analysis_text", ""),
    }

    with st.spinner("🤖 AI arbejder på at analysere input og oprette kampagnestruktur..."):
        system_prompt = """\
Du er en ekspert i Google Ads. Du skal hjælpe med at oprette en kampagnestruktur til en ny kunde.
Baseret på input fra en kravspecifikation (Xpect), en URL, analyse og ekstra noter, skal du foreslå følgende:
- Kampagnestruktur (Search / Display / evt. brand vs. generisk)
- Kampagnenavne
- Annoncegrupper (med tema og fokus)
- Søgeord (maks 10 pr. gruppe)
- Dagsbudget pr. kampagne
- En annonce pr. annoncegruppe med minimum 9 overskrifter og 4 beskrivelser
- Generér altid kampagneudvidelser: præcis 4 undersidelinks, 3-4 infotekster, structured snippets og opkaldsudvidelser.
Output skal være i JSON med topnøglen "campaigns".

**Vigtige retningslinjer for annoncetekster:**
- Skriv hele, men korte sætninger (mål 20–30 tegn). Afkort aldrig midt i et ord.
- Alle overskrifter skal være komplette sætninger der giver mening for brugeren.
- **Må IKKE slutte** på et enkelt funktionsord: "i", "på", "til", "for", "med", "om", "af", "hos", "ved", "uden", "over", "under", "mellem".
- Brug naturligt, flydende sprog som i professionelle Google Ads på dansk.
- Brug kun stort begyndelsesbogstav ved starten af sætning eller ved egennavne.
- Beskrivelser må være op til 90 tegn og skal være fulde, meningsfulde sætninger – aldrig afbrudte.
- Undgå gentagelser, tomme ord eller halve sætninger.

**Eksempler på gode overskrifter (under 30 tegn):**
- “Flyt nemt og sikkert”
- “Få gratis flyttetilbud”
- “Transport i hele Jylland”
- “Professionel flyttehjælp”

**Eksempler på gode beskrivelser:**
- “Vi tilbyder tryg og hurtig flytning i hele Danmark.”
- “Få et gratis tilbud og flyt uden besvær.”
- “Erfaren flyttehjælp til private og erhverv.”
"""

        analysis_for_prompt = st.session_state.get("analysis_text", "")
        competitor_analysis_for_prompt = st.session_state.get("competitor_analysis_text", "")
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
"""

        try:
            if not api_key:
                st.error("Indtast din OpenAI API-nøgle i sidebaren for at køre AI-analysen.")
                raise RuntimeError("Missing API key")
            
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.4
            )
            output_text = response.choices[0].message["content"]

            try:
                data = extract_json_from_text(output_text)
            except ValueError as e:
                st.error(str(e))
                data = None

            if not data:
                st.info("AI genererede ingen gyldig kampagnestruktur.")
                st.stop()

            # --- Hent søgevolumen/CPC/konkurrence for alle keywords ---
            all_keywords = []
            for c in data.get("campaigns", []):
                for ag in c.get("ad_groups", []) or []:
                    all_keywords.extend((ag.get("keywords") or []))
            unique_keywords = sorted(set([k.strip() for k in all_keywords if isinstance(k, str) and k.strip()]))

            metrics_map = {}
            df_kw = None
            if unique_keywords and gads_customer_id:
                st.subheader("🔎 Henter søgevolumen fra Keyword Planner…")
                metrics_map = fetch_keyword_metrics(unique_keywords, gads_customer_id)
                if metrics_map:
                    # Vis som tabel i UI
                    table_rows = []
                    for kw in unique_keywords:
                        m = metrics_map.get(kw.lower(), {})
                        table_rows.append({
                            "Keyword": kw,
                            "Monthly Searches": m.get("monthly", 0),
                            "Competition": m.get("competition", ""),
                            "Top of page bid (low)": m.get("cpc_low", 0.0),
                            "Top of page bid (high)": m.get("cpc_high", 0.0),
                        })
                    df_kw = pd.DataFrame(table_rows)
                    st.dataframe(df_kw, use_container_width=True)
                    # Add download button for keyword data
                    st.download_button(
                        label="Download søgeordsdata (CSV)",
                        data=df_kw.to_csv(index=False, sep="\t", encoding="utf-16", lineterminator="\n"),
                        file_name="keyword_volume_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Ingen søgevolumendata returneret – fortsætter uden filtrering.")
            else:
                st.info("Ingen keywords fundet eller mangler Google Ads customer ID – fortsætter uden søgevolumen.")

            rows = []
            # --- Helper functions for asset cleaning ---
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
                """Gør linjen læsbar uden at klippe ord og uden hængende funktionsord."""
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

                    # Support alternative shapes (e.g., single 'ad' object or 'rsa')
                    if not ad and isinstance(ad_group.get("ad"), dict):
                        ad = ad_group.get("ad")
                    if not ad and isinstance(ad_group.get("rsa"), dict):
                        ad = ad_group.get("rsa")

                    # Normalize to expected keys
                    ad = normalize_ad_obj(ad, customer_website)

                    ad_headlines = [clean_asset(ad.get(f"headline_{i}", ""), 30) for i in range(1, 10)]
                    ad_descriptions = [clean_asset(ad.get(f"description_{i}", ""), 90) for i in range(1,5)]
                    final_url = ad.get("final_url", customer_website or "")

                    # Prepare pinned headlines info
                    pinned_headlines = ad.get("pinned_headlines", {}) if isinstance(ad.get("pinned_headlines"), dict) else {}

                    base_row = {
                        "Campaign": campaign_name,
                        "Campaign Type": campaign_type,
                        "Campaign Status": "Enabled",
                        # Editor expects Budget/Budget type columns
                        "Budget": budget,
                        "Budget type": "Daily",
                        # Optional but helps Editor parsing
                        "Networks": "Google search",
                        "Languages": "Danish;English",

                        "Ad Group": ad_group_name,
                        "Ad Group Status": "Enabled",

                        # Disambiguators
                        "Ad type": "Responsive search ad",
                        "Criterion Type": "",  # blank for ad rows
                        "Status": "Enabled",

                        # Keyword fields left blank for ad rows
                        "Keyword": "",
                        "Keyword Match Type": "",
                        "Keyword Status": "",

                        # Ad assets
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

                    # Add rows for each keyword
                    for keyword in keywords:
                        kw_row = base_row.copy()
                        kw_row.update({
                            # Disambiguators for keyword rows
                            "Ad type": "",  # must be blank for keywords
                            "Criterion Type": "Exact",
                            "Status": "Enabled",

                            "Keyword": keyword,
                            "Keyword Match Type": "Exact",
                            "Keyword Status": "Enabled",

                            # Clear ad asset fields for keyword rows
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