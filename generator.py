from openai import OpenAI
import streamlit as st
import json
import pandas as pd
import io
import re
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Google Ads – Kampagne Generator", layout="wide")
st.title("Google Ads – Kampagne Generator")

st.header("1. Indsæt Xpect")
xpect_text = st.text_area("Kopier hele Xpect-teksten her")

st.header("2. Kundens website")
customer_website = st.text_input("Indtast URL til kundens website")

from urllib.parse import urljoin, urlparse

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
    default=["Search", "Display"]
)

st.header("5. Anden vigtig information")
additional_info = st.text_area("Skriv evt. yderligere info, som ikke står i Xpect")

# OpenAI API key input in sidebar
api_key = st.sidebar.text_input("Indtast din OpenAI API-nøgle", type="password")
# client is created later right before the API call to allow the user to change the key in the sidebar

# Optional: model selection in sidebar
model_choice = st.sidebar.selectbox("Vælg model", options=["gpt-4o", "gpt-4o-mini"], index=0)

def extract_json_from_text(text: str):
    """Try to extract a JSON object or array from a text blob.
    Returns Python object or raises ValueError.
    """
    # First try direct load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt to find the first JSON block using regex
    # Add a pattern to look for a top-level "campaigns" JSON object explicitly
    patterns = [r"\{.*\"campaigns\".*\}", r"\{.*\}", r"\[.*\]"]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            candidate = m.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                continue

    # Prøv at lukke afbrudt JSON ved at tilføje manglende afslutning
    if '"campaigns": [' in text and not text.strip().endswith("]}"):
        try:
            partial = text[: text.rfind("]") + 1] + "}"
            return json.loads(partial)
        except Exception:
            pass

    # If nothing worked, raise
    raise ValueError("Kunne ikke finde gyldigt JSON i AI-outputtet.")

def stringify_list(items):
    """Convert a list of strings or dicts to readable semicolon-separated string."""
    if not items:
        return ""
    result = []
    for item in items:
        if isinstance(item, dict):
            # fx {'text': 'Kontakt', 'url': 'https://...'}
            text = item.get("text") or item.get("title") or str(item)
            result.append(text)
        else:
            result.append(str(item))
    return "; ".join(result)

if st.button("Gem og fortsæt"):
    input_data = {
        "xpect": xpect_text,
        "website": customer_website,
        "extra": additional_info,
        "budget": total_daily_budget,
        "campaign_types": selected_campaign_types,
    }

    with st.spinner("🤖 AI arbejder på at analysere input og oprette kampagnestruktur..."):
        # Kald til OpenAI
        system_prompt = """\
Du er en ekspert i Google Ads. Du skal hjælpe med at oprette en kampagnestruktur til en ny kunde.
Baseret på input fra en kravspecifikation (Xpect), en URL og ekstra noter, skal du foreslå følgende:
- Kampagnestruktur (Search / Display / evt. brand vs. generisk)
- Kampagnenavne
- Annoncegrupper (med tema og fokus)
- Søgeord (maks 10 pr. gruppe)
- Dagsbudget pr. kampagne (baseret på forventet performance og målsætning)
- Et eksempel på én annonce pr. annoncegruppe med minimum 9 overskrifter og 4 beskrivelser
- Generér altid kampagneudvidelser: præcis 4 undersidelinks ("sitelinks"), 3-4 infotekster ("callouts"), structured snippets og opkaldsudvidelser. Hvis du er i tvivl om indhold, så find passende eksempler ud fra virksomhedstype og website. Disse skal bruges i eksporten, så alle fire typer kampagneudvidelser skal udfyldes for hver kampagne.

Output skal være i JSON-format og gerne startende med et top-level objekt med nøglen "campaigns".
Eksempelstruktur:
{
  "campaigns": [
    {
      "name": "...",
      "type": "Search",
      "daily_budget": 150,
      "sitelinks": [...],  # 4 undersidelinks
      "callouts": [...],   # 3-4 infotekster
      "structured_snippets": [...],
      "call_extensions": [...],
      "ad_groups": [
        {
          "name": "...",
          "keywords": ["..."],
          "ads": [
            {
              "headline_1": "...",
              "headline_2": "...",
              "headline_3": "...",
              "headline_4": "...",
              "headline_5": "...",
              "headline_6": "...",
              "headline_7": "...",
              "headline_8": "...",
              "headline_9": "...",
              "description_1": "...",
              "description_2": "...",
              "description_3": "...",
              "description_4": "...",
              "final_url": "https://..."
            }
          ]
        }
      ]
    }
  ]
}

Lav mange variationer i annonceformuleringer og søgeord. Hver kampagne bør have mindst 3-5 annoncegrupper, og hver annoncegruppe skal have kun én annonce med minimum 9 overskrifter og 4 beskrivelser. Brug både emotionelle, rationelle og USP-baserede vinkler. Brug scanning af websitet og information fra Xpect til at finde forskellige produktkategorier og vinkler til kampagnestrukturen. 

Overskrifter:
- Max tre overskrifter må indeholde et direkte søgeord – disse tre skal pinnes til Headline position 1
- Tre overskrifter skal være call to action (fx "Få et tilbud i dag", "Kontakt os nu") – de må ikke pinnes
- Tre overskrifter skal indeholde USP’er (fx "30 års erfaring", "Fast lav pris") – de må ikke pinnes
- Undgå dobbeltkonfekt og gentagelser i overskrifter og beskrivelser
- Brug kun stort begyndelsesbogstav ved starten af sætning eller ved navne/brands

Beskrivelser:
- Beskrivelser skal være varierede og supplerende til overskrifter – ikke gentage budskabet
"""
        user_prompt = f"""\
Xpect:
{xpect_text}

Website:
{customer_website}

Ekstra noter:
{additional_info}

Website-indhold fundet ved scanning:
{scraped_info}

Ønskede kampagnetyper: {", ".join(selected_campaign_types)}.
Brug kun disse i strukturen.

Samlet dagsbudget for alle kampagner må ikke overstige {total_daily_budget} kr.

Lav minimum 3 kampagner – fx baseret på forskellige produktkategorier, brands, eller kundetyper.

Hver kampagne bør have 3-5 annoncegrupper, og hver annoncegruppe bør have kun én annonce med minimum 9 overskrifter og 4 beskrivelser. Brug forskellige tekstvinkler: emotionel appel, rationelle fordele, produktfordele, unikke salgsargumenter og call-to-action variationer. Brug alle relevante input (Xpect, scanning, noter) til at generere rig variation.

Hver kampagne skal have mindst 2 annoncegrupper.
Brug mange variationer i søgeord og annoncetekster.
Tilføj 10 relevante søgeord pr. annoncegruppe.

Brug website-indholdet som ekstra inspiration til kampagnestrukturen og annoncetekster.

Tilføj altid kampagneudvidelser i outputtet: 4 undersidelinks ("sitelinks"), 3-4 infotekster ("callouts"), structured snippets og opkaldsudvidelser. Hvis du er i tvivl, så gæt ud fra Xpect og website. Alle fire udvidelsestyper skal være med i outputtet.
    """

        try:
            if not api_key:
                st.error("Indtast din OpenAI API-nøgle i sidebaren for at køre AI-analysen.")
                raise RuntimeError("Missing API key")
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.8
            )
            output_text = response.choices[0].message.content

            try:
                data = extract_json_from_text(output_text)
            except ValueError as e:
                st.error(str(e))
                data = None

            if data:
                rows = []
                for campaign in data.get("campaigns", []):
                    campaign_name = campaign.get("name", "")
                    campaign_type = campaign.get("type", "")
                    budget = campaign.get("daily_budget", "")

                    # Get campaign extensions (may be empty lists)
                    sitelinks = campaign.get("sitelinks", [])
                    callouts = campaign.get("callouts", [])
                    structured_snippets = campaign.get("structured_snippets", [])
                    call_extensions = campaign.get("call_extensions", [])

                    for ad_group in campaign.get("ad_groups", []):
                        ad_group_name = ad_group.get("name", "")
                        keywords = ad_group.get("keywords", []) or []
                        ads = ad_group.get("ads", []) or []

                        # Only one ad per ad group expected
                        ad = ads[0] if ads else {}

                        # Prepare ad fields with truncation and formatting
                        def trunc(text, length):
                            if not text:
                                return ""
                            text = text.strip()
                            return text[:length] if len(text) > length else text

                        def format_headline_case(text):
                            if not text:
                                return ""
                            text = text.strip()
                            words = text.split()
                            formatted = [words[0].capitalize()] if words else []
                            for word in words[1:]:
                                if word.lower() in ["i", "og", "eller", "for", "med", "på", "til", "af", "om", "fra", "ved", "uden"]:
                                    formatted.append(word.lower())
                                elif word[0].isupper() and word[1:].islower():
                                    formatted.append(word)  # Behold hvis det allerede er et navn/stednavn
                                else:
                                    formatted.append(word.lower())
                            return " ".join(formatted)

                        ad_headlines = [format_headline_case(trunc(ad.get(f"headline_{i}", ""), 30)) for i in range(1,10)]
                        ad_descriptions = [format_headline_case(trunc(ad.get(f"description_{i}", ""), 90)) for i in range(1,5)]
                        final_url = ad.get("final_url", customer_website or "")

                        # Add ad data row
                        row = {
                            "Campaign": campaign_name,
                            "Campaign Type": campaign_type,
                            "Ad Group": ad_group_name,
                            "Keyword": "",
                            "Headline 1": ad_headlines[0],
                            "Headline 2": ad_headlines[1],
                            "Headline 3": ad_headlines[2],
                            "Headline 4": ad_headlines[3],
                            "Headline 5": ad_headlines[4],
                            "Headline 6": ad_headlines[5],
                            "Headline 7": ad_headlines[6],
                            "Headline 8": ad_headlines[7],
                            "Headline 9": ad_headlines[8],
                            "Description 1": ad_descriptions[0],
                            "Description 2": ad_descriptions[1],
                            "Description 3": ad_descriptions[2],
                            "Description 4": ad_descriptions[3],
                            "Final URL": final_url,
                            "Budget": budget,
                            # Store campaign extensions for later use (if needed)
                            "Sitelinks": sitelinks,
                            "Callouts": callouts,
                            "Structured Snippets": structured_snippets,
                            "Call Extensions": call_extensions
                        }
                        rows.append(row)

                        # Add keyword rows
                        for keyword in keywords:
                            row_kw = {
                                "Campaign": campaign_name,
                                "Campaign Type": campaign_type,
                                "Ad Group": ad_group_name,
                                "Keyword": keyword,
                                "Headline 1": "",
                                "Headline 2": "",
                                "Headline 3": "",
                                "Headline 4": "",
                                "Headline 5": "",
                                "Headline 6": "",
                                "Headline 7": "",
                                "Headline 8": "",
                                "Headline 9": "",
                                "Description 1": "",
                                "Description 2": "",
                                "Description 3": "",
                                "Description 4": "",
                                "Final URL": customer_website or "",
                                "Budget": budget,
                                "Sitelinks": [],
                                "Callouts": [],
                                "Structured Snippets": [],
                                "Call Extensions": []
                            }
                            rows.append(row_kw)

                if rows:
                    ads_editor_columns = [
                        "Campaign", "Ad Group", "Ad type", "Campaign Type", "Networks", "Budget", "Budget type", "Languages",
                        "Campaign Status", "Ad Group Status", "Status", "Final URL", "Path 1", "Path 2",
                        "Headline 1", "Headline 1 Pinning", "Headline 2", "Headline 2 Pinning",
                        "Headline 3", "Headline 4", "Headline 5", "Headline 6", "Headline 7", "Headline 8", "Headline 9",
                        "Description 1", "Description 2", "Description 3", "Description 4", "Criterion Type", "Keyword",
                        "Sitelinks", "Callouts", "Structured Snippets", "Call Extensions"
                    ]

                    formatted_rows = []
                    for row in rows:
                        campaign_name = row.get("Campaign", "")
                        campaign_type = row.get("Campaign Type", "")
                        budget = row.get("Budget", "")
                        ad_group_name = row.get("Ad Group", "")
                        keyword = row.get("Keyword", "")
                        is_ad_row = bool(row.get("Headline 1") or row.get("Headline 2") or row.get("Description 1"))
                        is_keyword_row = bool(keyword)

                        if is_ad_row:
                            ad = {
                                "headline_1": row.get("Headline 1", ""),
                                "headline_2": row.get("Headline 2", ""),
                                "headline_3": row.get("Headline 3", ""),
                                "headline_4": row.get("Headline 4", ""),
                                "headline_5": row.get("Headline 5", ""),
                                "headline_6": row.get("Headline 6", ""),
                                "headline_7": row.get("Headline 7", ""),
                                "headline_8": row.get("Headline 8", ""),
                                "headline_9": row.get("Headline 9", ""),
                                "description_1": row.get("Description 1", ""),
                                "description_2": row.get("Description 2", ""),
                                "description_3": row.get("Description 3", ""),
                                "description_4": row.get("Description 4", ""),
                                "final_url": row.get("Final URL", customer_website or "")
                            }
                            formatted = {col: "" for col in ads_editor_columns}
                            formatted.update({
                                "Campaign": campaign_name,
                                "Ad Group": ad_group_name,
                                "Ad type": "Responsive search ad",
                                "Campaign Type": campaign_type,
                                "Networks": "Google search",
                                "Budget": budget,
                                "Budget type": "Daily",
                                "Languages": "Danish;English",
                                "Campaign Status": "Enabled",
                                "Ad Group Status": "Enabled",
                                "Status": "Enabled",
                                "Final URL": ad.get("final_url", customer_website or ""),
                                "Path 1": ad_group_name.lower().replace(" ", "")[:15],
                                "Path 2": campaign_name.lower().replace(" ", "")[:15],
                                "Headline 1": ad.get("headline_1", "")[:30],
                                "Headline 1 Pinning": "Pinned 1",
                                "Headline 2": ad.get("headline_2", "")[:30],
                                "Headline 2 Pinning": "Pinned 1",
                                "Headline 3": ad.get("headline_3", "")[:30],
                                "Headline 4": ad.get("headline_4", "")[:30],
                                "Headline 5": ad.get("headline_5", "")[:30],
                                "Headline 6": ad.get("headline_6", "")[:30],
                                "Headline 7": ad.get("headline_7", "")[:30],
                                "Headline 8": ad.get("headline_8", "")[:30],
                                "Headline 9": ad.get("headline_9", "")[:30],
                                "Description 1": ad.get("description_1", ""),
                                "Description 2": ad.get("description_2", ""),
                                "Description 3": ad.get("description_3", ""),
                                "Description 4": ad.get("description_4", ""),
                                "Sitelinks": stringify_list(row.get("Sitelinks", [])),
                                "Callouts": stringify_list(row.get("Callouts", [])),
                                "Structured Snippets": stringify_list(row.get("Structured Snippets", [])),
                                "Call Extensions": stringify_list(row.get("Call Extensions", []))
                            })
                            formatted_rows.append(formatted)

                        if is_keyword_row:
                            formatted = {col: "" for col in ads_editor_columns}
                            formatted.update({
                                "Campaign": campaign_name,
                                "Ad Group": ad_group_name,
                                "Campaign Type": campaign_type,
                                "Networks": "Google search",
                                "Budget": budget,
                                "Budget type": "Daily",
                                "Languages": "Danish;English",
                                "Campaign Status": "Enabled",
                                "Ad Group Status": "Enabled",
                                "Status": "Enabled",
                                "Final URL": customer_website or "",
                                "Criterion Type": "Exact",
                                "Keyword": keyword,
                                "Sitelinks": "",
                                "Callouts": "",
                                "Structured Snippets": "",
                                "Call Extensions": ""
                            })
                            formatted_rows.append(formatted)

                    df_ads = pd.DataFrame(formatted_rows, columns=ads_editor_columns)

                    csv_buffer = io.BytesIO()
                    # Ensure output includes campaign extensions as columns and uses correct lineterminator
                    df_ads.to_csv(csv_buffer, index=False, sep="\t", encoding="utf-16", lineterminator="\n")
                    csv_buffer.seek(0)

                    st.success("✅ Kampagnestruktur genereret! Download filen nedenfor.")
                    st.download_button(
                        label="Download CSV til Ads Editor",
                        data=csv_buffer,
                        file_name="ads_editor_upload.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("AI genererede ingen rækker til CSV.")
        except Exception as e:
            st.error(f"Fejl under AI-kald: {e}")