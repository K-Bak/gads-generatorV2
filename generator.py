import openai
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

scraped_info = ""
if customer_website:
    try:
        response = requests.get(customer_website, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)]
        links = [a.get_text(strip=True) for a in soup.find_all("a") if a.get_text(strip=True)]

        scraped_info = "\n".join(headings[:10] + links[:10])
        st.write("🔎 Website analyseret – fundet følgende indhold:")
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
openai.api_key = st.sidebar.text_input("Indtast din OpenAI API-nøgle", type="password")

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
    - Et eksempel på en annonce pr. annoncegruppe

    Output skal være i JSON-format og gerne startende med et top-level objekt med nøglen "campaigns".
    Eksempelstruktur:
    {
      "campaigns": [
        {
          "name": "...",
          "type": "Search",
          "daily_budget": 150,
          "ad_groups": [
            {
              "name": "...",
              "keywords": ["..."],
              "ads": [
                {
                  "headline_1": "...",
                  "headline_2": "...",
                  "description": "...",
                  "final_url": "https://..."
                }
              ]
            }
          ]
        }
      ]
    }
    
    Lav mange variationer i annonceformuleringer og søgeord. Hver kampagne bør have mindst 3-5 annoncegrupper, og hver annoncegruppe skal have mindst 5-8 annoncer med forskellige formuleringer. Brug både emotionelle, rationelle og USP-baserede vinkler. Brug scanning af websitet og information fra Xpect til at finde forskellige produktkategorier og vinkler til kampagnestrukturen. 
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

Hver kampagne bør have 3-5 annoncegrupper, og hver annoncegruppe bør have 5-8 unikke annoncer. Brug forskellige tekstvinkler: emotionel appel, rationelle fordele, produktfordele, unikke salgsargumenter og call-to-action variationer. Brug alle relevante input (Xpect, scanning, noter) til at generere rig variation.

Hver kampagne skal have mindst 2 annoncegrupper.
Hver annoncegruppe skal have mindst 3 annoncer (headline 1+2 og description).
Brug mange variationer i søgeord og annoncetekster.
Tilføj 10 relevante søgeord pr. annoncegruppe.

Brug website-indholdet som ekstra inspiration til kampagnestrukturen og annoncetekster.
    """

        try:
            response = openai.ChatCompletion.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.8
            )
            output_text = response.choices[0].message["content"]

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

                    for ad_group in campaign.get("ad_groups", []):
                        ad_group_name = ad_group.get("name", "")
                        keywords = ad_group.get("keywords", []) or []
                        ads = ad_group.get("ads", []) or []

                        for ad in ads:
                            row = {
                                "Campaign": campaign_name,
                                "Campaign Type": campaign_type,
                                "Ad Group": ad_group_name,
                                "Keyword": "",
                                "Headline 1": ad.get("headline_1", "")[:30],
                                "Headline 2": ad.get("headline_2", "")[:30],
                                "Description": ad.get("description", ""),
                                "Final URL": ad.get("final_url", customer_website or ""),
                                "Budget": budget
                            }
                            rows.append(row)

                        for keyword in keywords:
                            ad = ads[0] if ads else {}
                            row = {
                                "Campaign": campaign_name,
                                "Campaign Type": campaign_type,
                                "Ad Group": ad_group_name,
                                "Keyword": keyword,
                                "Headline 1": ad.get("headline_1", "")[:30],
                                "Headline 2": ad.get("headline_2", "")[:30],
                                "Description": ad.get("description", ""),
                                "Final URL": ad.get("final_url", customer_website or ""),
                                "Budget": budget
                            }
                            rows.append(row)

                if rows:
                    ads_editor_columns = [
                        "Campaign", "Ad Group", "Ad type", "Campaign Type", "Networks", "Budget", "Budget type", "Languages",
                        "Campaign Status", "Ad Group Status", "Status", "Final URL", "Path 1", "Path 2",
                        "Headline 1", "Headline 1 Pinning", "Headline 2", "Headline 2 Pinning",
                        "Headline 3", "Headline 4", "Headline 5", "Headline 6",
                        "Description 1", "Description 2", "Criterion Type", "Keyword"
                    ]

                    formatted_rows = []
                    for row in rows:
                        campaign_name = row.get("Campaign", "")
                        campaign_type = row.get("Campaign Type", "")
                        budget = row.get("Budget", "")
                        ad_group_name = row.get("Ad Group", "")
                        keyword = row.get("Keyword", "")
                        is_ad_row = bool(row.get("Headline 1") or row.get("Headline 2") or row.get("Description"))
                        is_keyword_row = bool(keyword)

                        if is_ad_row:
                            ad = {
                                "headline_1": row.get("Headline 1", ""),
                                "headline_2": row.get("Headline 2", ""),
                                "description": row.get("Description", ""),
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
                                "Headline 3": "Få et godt tilbud"[:30],
                                "Headline 4": "",
                                "Headline 5": "",
                                "Headline 6": "",
                                "Description 1": ad.get("description", ""),
                                "Description 2": "Kontakt os i dag for et tilbud."
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
                                "Keyword": keyword
                            })
                            formatted_rows.append(formatted)

                    df_ads = pd.DataFrame(formatted_rows, columns=ads_editor_columns)

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
                else:
                    st.info("AI genererede ingen rækker til CSV.")
        except Exception as e:
            st.error(f"Fejl under AI-kald: {e}")