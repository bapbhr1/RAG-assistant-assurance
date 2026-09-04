# Interface Streamlit de l'assistant contrats.

from __future__ import annotations

import html
import os
import re
import uuid
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.chunking import build_corpus, load_contracts
from src.models import ConversationTurn, CoverageStatus, RAGQuery, RAGResponse
from src.rag import DEFAULT_GROQ_MODEL, RAGEngine
from src.retriever import HybridRetriever

DATA_PATH = Path(__file__).parent / "data" / "synthetic_contracts.json"
HERO_PATH = Path(__file__).parent / "assets" / "contract-assistant-hero.png"
APP_STATE_VERSION = 5

PRODUCT_LINES = {
    "Toutes les branches": None,
    "Automobile": "auto",
    "Moto": "moto",
    "Habitation": "habitation",
    "Santé": "sante",
    "Prévoyance": "prevoyance",
    "Accidents de la vie (GAV)": "gav",
    "Emprunteur": "emprunteur",
    "Voyage": "voyage",
    "Animaux": "animaux",
    "Obsèques": "obseques",
    "Scolaire": "scolaire",
    "Professionnelle": "pro",
}
STATUS = {
    CoverageStatus.COVERED: ("Pris en charge", "status-covered"),
    CoverageStatus.NOT_COVERED: ("Non pris en charge", "status-declined"),
    CoverageStatus.PARTIAL: ("Prise en charge partielle", "status-partial"),
    CoverageStatus.NEEDS_REVIEW: ("Décision sur le dossier à confirmer", "status-review"),
}
EXAMPLES = [
    "Suis-je couvert pour un vol annulé suite à une maladie ?",
    "Quelle franchise appliquer pour un dégât des eaux dans mon logement ?",
    "Quel remboursement est prévu pour des prothèses dentaires ?",
    "L'arrêt de travail est-il couvert par l'assurance emprunteur ?",
]
CHAT_VIEW = "Assistant"
DOCUMENTS_VIEW = "Documents"


def _configure_network() -> None:
    # reporte un éventuel proxy d'entreprise (secrets) vers les variables d'env
    try:
        proxy = st.secrets.get("HTTPS_PROXY")
    except Exception:
        proxy = None
    if proxy:
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)


def get_api_key() -> str | None:
    # secrets d'abord, puis variable d'environnement
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secret_key = None
    return secret_key or os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner="Préparation des contrats…")
def get_retriever() -> HybridRetriever:
    retriever = HybridRetriever()
    retriever.index(build_corpus(load_contracts(DATA_PATH)))
    return retriever


@st.cache_data
def get_contracts() -> list[dict]:
    return [contract.model_dump() for contract in load_contracts(DATA_PATH)]


def confidence_label(response: RAGResponse) -> tuple[str, str]:
    if response.confidence >= 0.85:
        if response.status == CoverageStatus.NEEDS_REVIEW:
            has_supported_information = any(
                check.supported for check in response.evidence_checks
            )
            if has_supported_information:
                return (
                    "Élevée",
                    "La réponse sur le contrat est vérifiée. Une validation reste nécessaire pour l'appliquer à ce dossier.",
                )
            return (
                "Élevée",
                "L'assistant a correctement identifié qu'il ne pouvait pas conclure avec les éléments disponibles.",
            )
        return "Élevée", "La réponse et ses informations ont passé les contrôles prévus."
    if response.confidence >= 0.65:
        return "Modérée", "La réponse est prudente, mais certains éléments sont moins solidement confirmés."
    return "Faible", "Une ou plusieurs informations de la réponse n'ont pas pu être confirmées."


def _open_article(contract_id: str, article_id: str) -> None:
    st.session_state.active_view = DOCUMENTS_VIEW
    st.session_state.focus_contract_id = contract_id
    st.session_state.focus_article_id = article_id


def _article_button(contract_id: str, article_id: str, title: str, key: str) -> None:
    st.button(
        f"{contract_id} · {article_id} — {title}",
        key=key,
        on_click=_open_article,
        args=(contract_id, article_id),
    )


def _notice(message: str, tone: str = "info") -> None:
    st.markdown(
        f"<div class='notice notice-{tone}'>{html.escape(message)}</div>",
        unsafe_allow_html=True,
    )


def render_response(response: RAGResponse, key_prefix: str = "response") -> None:
    label, status_class = STATUS[response.status]
    st.markdown(
        f"<div class='status-pill {status_class}'>{html.escape(label)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Réponse au gestionnaire")
    st.write(response.answer)

    st.markdown(
        "<div class='decision-card'><span>DÉCISION RECOMMANDÉE</span><br>"
        f"<strong>{html.escape(response.decision)}</strong></div>",
        unsafe_allow_html=True,
    )

    if response.conditions:
        st.markdown("**Conditions et limites à communiquer**")
        for condition in response.conditions:
            st.markdown(f"- {condition}")
    if response.missing_information:
        st.markdown("**À confirmer pour appliquer la garantie au dossier**")
        for item in response.missing_information:
            st.markdown(f"- {item}")

    conf_name, conf_help = confidence_label(response)
    left, right = st.columns([1, 3], vertical_alignment="center")
    left.metric("Fiabilité de la réponse", f"{response.confidence:.0%}", conf_name)
    with right:
        st.progress(response.confidence)
        st.caption(conf_help)

    with st.expander("Voir le calcul de la fiabilité"):
        breakdown = response.confidence_breakdown
        facts_points = round(breakdown.factual_support * 50)
        caution_points = round(breakdown.uncertainty_handling * 30)
        sources_points = round(breakdown.source_handling * 20)
        total_points = facts_points + caution_points + sources_points
        checks = response.evidence_checks
        supported = sum(check.supported for check in checks)

        st.markdown(
            f"### {facts_points}/50 + {caution_points}/30 + "
            f"{sources_points}/20 = {total_points}/100"
        )
        if checks:
            st.write(
                f"**Informations vérifiées — {facts_points}/50**  \n"
                f"{supported} affirmation(s) sur {len(checks)} retrouvée(s) dans les contrats."
            )
            for check in checks:
                check_label = "Confirmé" if check.supported else "À vérifier"
                st.write(f"**{check_label} —** {check.claim}")
        else:
            no_claim_message = (
                "L'assistant n'a présenté aucune affirmation contractuelle non vérifiée."
                if facts_points == 50
                else "Aucune réponse exploitable n'a pu être contrôlée."
            )
            st.write(f"**Informations vérifiées — {facts_points}/50**  \n{no_claim_message}")
        st.write(
            f"**Réponse adaptée — {caution_points}/30**  \n"
            + (
                "L'assistant a répondu ou demandé une validation au bon moment."
                if caution_points == 30
                else "L'assistant a été trop affirmatif ou trop prudent au regard des éléments disponibles."
            )
        )
        st.write(
            f"**Utilisation des sources — {sources_points}/20**  \n"
            + (
                "Les sources utiles ont été exploitées, ou leur insuffisance a été correctement signalée."
                if sources_points == 20
                else "Le lien entre la question et les sources disponibles est moins direct."
            )
        )
        st.caption(
            "Une fiabilité élevée peut accompagner une absence de réponse : cela signifie alors "
            "que l'assistant a correctement reconnu qu'il ne devait pas conclure. "
            "Cet indice mesure les contrôles automatiques, pas une certitude juridique."
        )

    st.markdown("#### Preuves contractuelles")
    if not response.citations:
        _notice(
            "Aucun extrait contractuel suffisamment fiable ne permet d'appuyer cette réponse.",
            "warning",
        )
    grouped_citations: dict[tuple[str, str, str], list[str]] = {}
    for citation in response.citations:
        key = (citation.contract_id, citation.article_id, citation.article_title)
        if citation.quote not in grouped_citations.setdefault(key, []):
            grouped_citations[key].append(citation.quote)
    for (contract_id, article_id, article_title), quotes in grouped_citations.items():
        with st.expander(
            f"{contract_id} · {article_id} — {article_title}",
            expanded=True,
        ):
            for quote in quotes:
                st.markdown(f"> {quote}")
            st.caption(f"Extrait de {contract_id}, article {article_id}.")
            _article_button(
                contract_id,
                article_id,
                article_title,
                key=f"{key_prefix}_proof_{contract_id}_{article_id}",
            )

    with st.expander("Pourquoi cette réponse ?"):
        st.markdown("**Règle appliquée au dossier**")
        st.write(response.reasoning_summary)
        st.markdown("**Clauses prises en compte**")
        seen_articles: set[tuple[str, str]] = set()
        for index, item in enumerate(response.retrieved):
            article_key = (item.chunk.contract_id, item.chunk.article_id)
            if article_key in seen_articles:
                continue
            seen_articles.add(article_key)
            _article_button(
                item.chunk.contract_id,
                item.chunk.article_id,
                item.chunk.article_title,
                key=f"{key_prefix}_reason_{index}_{item.chunk.contract_id}_{item.chunk.article_id}",
            )
        if response.warnings:
            _notice(
                "Certains éléments n'ont pas pu être confirmés dans les contrats. "
                "Vérifiez le dossier avant de communiquer une décision au client.",
                "warning",
            )


def _create_conversation() -> None:
    conversation_id = uuid.uuid4().hex
    discussion_number = len(st.session_state.conversations) + 1
    st.session_state.conversations[conversation_id] = {
        "title": f"Discussion {discussion_number}",
        "messages": [],
    }
    st.session_state.active_conversation_id = conversation_id
    st.session_state.active_view = CHAT_VIEW


def _ensure_active_conversation() -> tuple[str, dict]:
    conversation_id = st.session_state.get("active_conversation_id")
    conversations = st.session_state.conversations
    if conversation_id not in conversations:
        _create_conversation()
        conversation_id = st.session_state.active_conversation_id
    return conversation_id, conversations[conversation_id]


def _switch_conversation(conversation_id: str) -> None:
    st.session_state.active_conversation_id = conversation_id
    st.session_state.active_view = CHAT_VIEW


def _use_example(question: str, widget_key: str) -> None:
    st.session_state[widget_key] = question


def render_conversation_menu() -> None:
    with st.popover("", icon=":material/forum:"):
        st.markdown(
            '<div class="discussion-panel-marker"></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Discussions")
        st.button(
            "Nouvelle discussion",
            icon=":material/add:",
            width="stretch",
            on_click=_create_conversation,
            key="new_conversation",
        )
        st.caption("Les discussions sont conservées pendant cette session.")
        st.divider()
        active_id = st.session_state.active_conversation_id
        for conversation_id, conversation in reversed(
            list(st.session_state.conversations.items())
        ):
            st.button(
                conversation["title"],
                key=f"conversation_{conversation_id}",
                type="primary" if conversation_id == active_id else "secondary",
                width="stretch",
                on_click=_switch_conversation,
                args=(conversation_id,),
            )


def render_chat(api_key: str | None) -> None:
    conversation_id, conversation = _ensure_active_conversation()
    messages = conversation["messages"]
    question_key = f"question_{conversation_id}"
    branch_key = f"branch_{conversation_id}"

    if st.session_state.pop("clear_question_key", None) == question_key:
        st.session_state[question_key] = ""

    st.subheader("Assistant de décision sinistre")
    st.caption(
        "Poursuivez la discussion : l'assistant tient compte des échanges précédents "
        "tout en vérifiant chaque information dans les contrats."
    )
    branch = st.selectbox(
        "Type de contrat à consulter", list(PRODUCT_LINES), key=branch_key
    )

    if not messages:
        st.markdown("**Exemples de questions**")
        columns = st.columns(2)
        for index, example in enumerate(EXAMPLES):
            columns[index % 2].button(
                example,
                key=f"example_{conversation_id}_{index}",
                width="stretch",
                on_click=_use_example,
                args=(example, question_key),
            )

    for index, message in enumerate(messages):
        is_latest = index == len(messages) - 1
        st.markdown(
            "<div class='question-card'><span>Votre question</span>"
            f"<p>{html.escape(message['question'])}</p></div>",
            unsafe_allow_html=True,
        )
        if is_latest:
            st.markdown(
                '<div id="latest-response" class="response-anchor"></div>',
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            st.markdown(
                "<div class='response-label'>Réponse de l'assistant</div>",
                unsafe_allow_html=True,
            )
            render_response(
                message["response"], key_prefix=f"{conversation_id}_{index}"
            )

    with st.form(f"question_form_{conversation_id}"):
        question = st.text_area(
            "Votre prochaine question" if messages else "Situation ou question personnalisée",
            key=question_key,
            placeholder=(
                "Exemple : et quelle franchise s'applique dans ce cas ?"
                if messages
                else "Décrivez le sinistre et les informations déjà disponibles…"
            ),
            height=100,
        )
        submitted = st.form_submit_button(
            "Envoyer la question" if messages else "Analyser le dossier",
            type="primary",
            disabled=not api_key,
        )

    if not api_key:
        _notice(
            "L'assistant est momentanément indisponible. Contactez l'administrateur de l'application."
        )
    if submitted:
        if len(question.strip()) < 3:
            _notice("Décrivez le dossier en quelques mots.", "warning")
        else:
            history = [
                ConversationTurn(
                    question=message["question"],
                    answer=message["response"].answer,
                )
                for message in messages[-4:]
            ]
            try:
                with st.spinner("Recherche des clauses, analyse et contrôle des preuves…"):
                    engine = RAGEngine(
                        get_retriever(), api_key=api_key, model=DEFAULT_GROQ_MODEL
                    )
                    response = engine.answer(
                        RAGQuery(
                            question=question.strip(),
                            product_line=PRODUCT_LINES[branch],
                            top_k=st.session_state.top_k,
                            conversation_history=history,
                        )
                    )
                is_first_message = not messages
                messages.append({"question": question.strip(), "response": response})
                if is_first_message:
                    conversation["title"] = (
                        question.strip()[:46] + ("…" if len(question.strip()) > 46 else "")
                    )
                st.session_state.clear_question_key = question_key
                st.session_state.scroll_conversation_id = conversation_id
                st.rerun()
            except Exception:
                _notice("L'analyse n'a pas pu aboutir. Réessayez dans quelques instants.", "error")

    if st.session_state.get("scroll_conversation_id") == conversation_id:
        st.html(
            """<script>
            window.setTimeout(() => {
              const target = document.getElementById('latest-response');
              if (target) target.scrollIntoView({behavior: 'smooth', block: 'start'});
            }, 180);
            </script>""",
            unsafe_allow_javascript=True,
        )
        st.session_state.scroll_conversation_id = None


def _highlight(text: str, search: str) -> str:
    # surligne le terme recherché dans un texte déjà échappé
    safe = html.escape(text)
    if not search:
        return safe
    return re.sub(
        re.escape(html.escape(search)),
        lambda match: f"<mark>{match.group(0)}</mark>",
        safe,
        flags=re.IGNORECASE,
    )


def render_documents() -> None:
    contracts = get_contracts()
    articles = sum(len(contract["articles"]) for contract in contracts)
    branches = len({contract["product_line"] for contract in contracts})
    st.subheader("Portefeuille documentaire fictif")
    m1, m2, m3 = st.columns(3)
    m1.metric("Contrats", len(contracts))
    m2.metric("Clauses disponibles", articles)
    _notice("Jeu de données entièrement synthétique : aucune donnée client réelle n'est utilisée.")

    focus_contract_id = st.session_state.get("focus_contract_id")
    focus_article_id = st.session_state.get("focus_article_id")
    if focus_contract_id and focus_article_id:
        target_contract = next(
            (contract for contract in contracts if contract["contract_id"] == focus_contract_id),
            None,
        )
        target_article = next(
            (
                article
                for article in (target_contract or {}).get("articles", [])
                if article["article_id"] == focus_article_id
            ),
            None,
        )
        if target_contract and target_article:
            _notice(
                f"Article ouvert depuis la réponse : {focus_contract_id} · "
                f"{focus_article_id} — {target_article['title']}",
                "success",
            )
            if st.button("Revenir à tous les documents"):
                st.session_state.focus_contract_id = None
                st.session_state.focus_article_id = None
                st.rerun()
            st.markdown(f"### {target_contract['product']} — {target_contract['title']}")
            st.caption(
                f"Contrat {target_contract['contract_id']} · "
                f"Souscripteur fictif : {target_contract['policyholder']}"
            )
            st.markdown(
                "<div class='focused-clause'>"
                f"<strong>{html.escape(target_article['article_id'])} · "
                f"{html.escape(target_article['title'])}</strong><br>"
                f"{html.escape(target_article['text'])}</div>",
                unsafe_allow_html=True,
            )
            return

        st.session_state.focus_contract_id = None
        st.session_state.focus_article_id = None

    counts = (
        pd.DataFrame(
            [{"Branche": contract["product"], "Clauses": len(contract["articles"])} for contract in contracts]
        )
        .groupby("Branche", as_index=False)["Clauses"]
        .sum()
    )
    st.markdown("#### Répartition des clauses par produit")
    palette = [
        "#4285F4",
        "#5B78F1",
        "#746BEA",
        "#8E5EE8",
        "#796FEC",
        "#538FDB",
        "#23B8D8",
    ]
    branch_colors = [palette[index % len(palette)] for index in range(len(counts))]
    color_scale = alt.Scale(
        domain=counts["Branche"].tolist(),
        range=branch_colors,
    )
    bars = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=8, height=23)
        .encode(
            y=alt.Y(
                "Branche:N",
                sort=alt.SortField(field="Clauses", order="descending"),
                title=None,
                axis=alt.Axis(labelColor="#59636E", labelPadding=10, ticks=False, domain=False),
            ),
            x=alt.X(
                "Clauses:Q",
                title=None,
                axis=alt.Axis(labels=False, ticks=False, domain=False, grid=False),
                scale=alt.Scale(domain=[0, int(counts["Clauses"].max()) + 1]),
            ),
            color=alt.Color("Branche:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("Branche:N", title="Branche"),
                alt.Tooltip("Clauses:Q", title="Clauses"),
            ],
        )
    )
    labels = bars.mark_text(
        align="left", baseline="middle", dx=8, color="#59636E", fontWeight=600
    ).encode(text=alt.Text("Clauses:Q"))
    chart = (
        (bars + labels)
        .properties(height=270)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, labelFontSize=12)
    )
    st.altair_chart(chart, width="stretch")

    left, right = st.columns([2, 1])
    search = left.text_input("Rechercher une clause", placeholder="vol, franchise, carence, plafond…").strip()
    branch_label = right.selectbox("Filtrer par branche", list(PRODUCT_LINES), key="document_branch")
    branch_filter = PRODUCT_LINES[branch_label]

    displayed = 0
    for contract in contracts:
        if branch_filter and contract["product_line"] != branch_filter:
            continue
        matching = [
            article for article in contract["articles"]
            if not search or search.casefold() in f"{article['title']} {article['text']}".casefold()
        ]
        if not matching:
            continue
        displayed += len(matching)
        with st.expander(
            f"{contract['product']} — {contract['title']} · {contract['contract_id']}",
            expanded=bool(search),
        ):
            st.caption(f"Souscripteur fictif : {contract['policyholder']}")
            for article in matching:
                st.markdown(
                    f"<div class='clause'><strong>{html.escape(article['article_id'])} · "
                    f"{_highlight(article['title'], search)}</strong><br>"
                    f"{_highlight(article['text'], search)}</div>",
                    unsafe_allow_html=True,
                )


def main() -> None:
    _configure_network()
    st.set_page_config(
        page_title="Assistant IA pour contrats",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if st.session_state.get("app_state_version") != APP_STATE_VERSION:
        st.session_state.conversations = {}
        st.session_state.active_conversation_id = None
        st.session_state.active_view = CHAT_VIEW
        st.session_state.focus_contract_id = None
        st.session_state.focus_article_id = None
        st.session_state.app_state_version = APP_STATE_VERSION
    st.session_state.setdefault("top_k", 5)
    st.session_state.setdefault("active_view", CHAT_VIEW)
    st.session_state.setdefault("focus_contract_id", None)
    st.session_state.setdefault("focus_article_id", None)
    st.session_state.setdefault("conversations", {})
    st.session_state.setdefault("active_conversation_id", None)
    st.session_state.setdefault("scroll_conversation_id", None)
    if not st.session_state.conversations:
        _create_conversation()
    st.markdown(
        """<style>
        :root {
          --ink:#202124;--muted:#687078;--border:#e2e7f0;--surface:#ffffff;
          --soft:#f5f7ff;--blue:#4285f4;--violet:#8e5ee8;--cyan:#23b8d8;
        }
        .stApp {
          background:
            radial-gradient(circle at 82% 2%,rgba(142,94,232,.10),transparent 24rem),
            radial-gradient(circle at 14% 22%,rgba(35,184,216,.08),transparent 22rem),
            #f8faff;
          color:var(--ink);
        }
        [data-testid="stHeader"] {height:2.75rem;background:transparent;}
        [data-testid="stToolbar"] {height:2.75rem;}
        .block-container {max-width:1120px;padding-top:2.4rem;padding-bottom:4rem;}
        h1,h2,h3,h4 {color:var(--ink);letter-spacing:-.025em;}
        p {line-height:1.6;}
        .hero-copy {padding:.6rem 0 1.2rem;}
        .hero-kicker {font-size:.76rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
          background:linear-gradient(90deg,var(--blue),var(--violet),var(--cyan));
          -webkit-background-clip:text;color:transparent;margin-bottom:.55rem;}
        .hero-title {font-size:2.45rem;font-weight:680;line-height:1.08;margin:0 0 .75rem;color:var(--ink);}
        .hero-subtitle {font-size:1.02rem;color:var(--muted);max-width:610px;margin:0;}
        [data-testid="stImage"] {display:flex;justify-content:center;align-items:center;}
        [data-testid="stImage"] img {
          width:92%!important;max-width:330px;max-height:220px;object-fit:contain;
          margin:0 auto;transform:translateY(28px);
        }
        [data-testid="stSidebar"] {background:#fbfcff;border-right:1px solid var(--border);}
        [data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
        [data-testid="stForm"], [data-testid="stMetric"] {
          background:rgba(255,255,255,.92);border:1px solid var(--border);border-radius:16px;
          box-shadow:0 6px 22px rgba(52,72,120,.055);
        }
        [data-testid="stForm"] {padding:1.25rem 1.3rem .35rem;}
        [data-testid="stMetric"] {padding:1rem 1.1rem;min-height:112px;}
        [data-testid="stVerticalBlockBorderWrapper"] {background:rgba(255,255,255,.94)!important;
          border:1px solid var(--border)!important;border-radius:18px!important;
          box-shadow:0 8px 26px rgba(52,72,120,.06);padding:.35rem;}
        [data-testid="stExpander"] {background:rgba(255,255,255,.92);border:1px solid var(--border);
          border-radius:14px;box-shadow:none;overflow:hidden;}
        [data-testid="stSegmentedControl"] {background:rgba(255,255,255,.86);border:1px solid var(--border);
          border-radius:14px;padding:4px;width:fit-content;margin-bottom:0;}
        /* Les deux commandes restent dans les coins de la fenêtre au défilement. */
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"] {
          position:fixed!important;top:.55rem!important;left:.7rem!important;z-index:1000001;
          width:6rem!important;height:2.65rem;overflow:visible!important;
        }
        [data-testid="stExpandSidebarButton"] button,
        [data-testid="stSidebarCollapseButton"] button {
          width:6rem!important;min-width:6rem!important;height:2.65rem;padding:0!important;
          display:flex!important;align-items:center!important;justify-content:center!important;
          border:1px solid #d9e0ee;border-radius:11px;background:rgba(255,255,255,.96);
          box-shadow:0 3px 12px rgba(52,72,120,.09);
        }
        [data-testid="stExpandSidebarButton"] button > *,
        [data-testid="stExpandSidebarButton"] > svg,
        [data-testid="stExpandSidebarButton"] > span,
        [data-testid="stSidebarCollapseButton"] button > *,
        [data-testid="stSidebarCollapseButton"] > svg,
        [data-testid="stSidebarCollapseButton"] > span {display:none!important;}
        [data-testid="stExpandSidebarButton"]::after,
        [data-testid="stSidebarCollapseButton"]::after {
          content:"help";display:block!important;position:absolute;left:1.35rem;top:50%;
          transform:translate(-50%,-50%);pointer-events:none;
          font-family:"Material Symbols Rounded";
          font-size:1.25rem;font-weight:400;font-style:normal;line-height:1;color:var(--ink);
          font-variation-settings:"FILL" 0,"wght" 400,"GRAD" 0,"opsz" 24;
        }
        [data-testid="stExpandSidebarButton"]::before,
        [data-testid="stSidebarCollapseButton"]::before {
          content:"Aide";display:block!important;position:absolute;left:2.55rem;top:50%;
          transform:translateY(-50%);pointer-events:none;color:var(--ink);
          font-size:1rem;font-weight:400;line-height:1;
        }
        [data-testid="stPopover"] {
          position:fixed!important;top:.55rem!important;right:.7rem!important;z-index:1000001;
          width:2.75rem!important;margin:0!important;
        }
        [data-testid="stPopover"] button {
          width:2.75rem!important;min-width:2.75rem!important;padding:0!important;
          display:flex!important;align-items:center!important;justify-content:center!important;
          min-height:2.65rem;border:1px solid #d9e0ee;border-radius:11px;
          background:rgba(255,255,255,.96);color:var(--ink)!important;font-weight:600;
          box-shadow:0 3px 12px rgba(52,72,120,.09);
        }
        [data-testid="stPopover"] button [data-testid="stMarkdownContainer"],
        [data-testid="stPopover"] button p {display:none!important;}
        /* Le contenu du popover devient un panneau latéral droit. */
        div[data-baseweb="popover"]:has(.discussion-panel-marker) {
          position:fixed!important;inset:0 0 0 auto!important;transform:none!important;
          width:min(23rem,92vw)!important;height:100dvh!important;max-height:100dvh!important;
          border-radius:18px 0 0 18px!important;border:0!important;
          border-left:1px solid var(--border)!important;background:#fbfcff!important;
          box-shadow:-14px 0 42px rgba(31,49,91,.16)!important;
          overflow-y:auto!important;
        }
        div[data-baseweb="popover"]:has(.discussion-panel-marker) > div {
          width:100%!important;max-width:none!important;max-height:none!important;
          border-radius:inherit!important;background:#fbfcff!important;
        }
        div[data-baseweb="popover"]:has(.discussion-panel-marker) [data-testid="stPopoverBody"] {
          width:100%!important;max-width:none!important;padding:1.25rem!important;
        }
        .nav-spacer {height:.9rem;}
        .stButton > button, [data-testid="stFormSubmitButton"] > button {
          border-radius:11px;border:1px solid #d9e0ee;min-height:2.65rem;font-weight:600;
          transition:transform .12s ease,box-shadow .12s ease,border-color .12s ease;
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
          border-color:#7da7f8;box-shadow:0 5px 14px rgba(66,133,244,.13);transform:translateY(-1px);
        }
        [data-testid="stFormSubmitButton"] > button {
          border:0;color:white;background:linear-gradient(100deg,var(--blue),var(--violet));
        }
        textarea, input, [data-baseweb="select"] > div {
          border-radius:11px!important;border-color:#d9e0ee!important;background:#fff!important;
        }
        hr {border-color:var(--border)!important;}
        .question-card {max-width:78%;margin:0 0 1rem auto;padding:.9rem 1.05rem;
          background:linear-gradient(110deg,#edf5ff,#f3efff);border:1px solid #d9e3f7;
          border-radius:16px 16px 4px 16px;color:var(--ink);}
        .question-card span,.response-label {font-size:.71rem;text-transform:uppercase;
          letter-spacing:.08em;font-weight:700;color:#6672a4;}
        .question-card p {margin:.35rem 0 0;line-height:1.55;}
        .response-label {margin:.15rem 0 .7rem;}
        .response-anchor {scroll-margin-top:3.5rem;}
        .notice {padding:.78rem .95rem;border-radius:12px;border:1px solid var(--border);
          background:#f4f7ff;color:#475569;font-size:.9rem;margin:.7rem 0;}
        .notice-success {background:#edf8f6;border-color:#c9ebe4;color:#256a60;}
        .notice-warning {background:#fff8ea;border-color:#f4dfb4;color:#795716;}
        .notice-error {background:#fff1f3;border-color:#f4ccd3;color:#9b3346;}
        .status-pill {display:inline-flex;align-items:center;padding:.48rem .8rem;border-radius:999px;
          font-size:.82rem;font-weight:700;margin:.15rem 0 .8rem;border:1px solid transparent;}
        .status-covered {background:#e6f6ec;color:#1a7f45;border-color:#bfe6cd;}
        .status-declined {background:#fff0f2;color:#b3263b;border-color:#ffd2d9;}
        .status-partial {background:#fff6e5;color:#8b5b00;border-color:#ffe1a8;}
        .status-review {background:#f2edff;color:#6842b8;border-color:#ded0ff;}
        .decision-card {position:relative;background:linear-gradient(110deg,#f2f7ff,#f7f2ff);
          border:1px solid #dbe4f6;border-radius:15px;padding:1rem 1.2rem;margin:1rem 0 1.2rem;}
        .decision-card::before {content:"";position:absolute;left:0;top:14px;bottom:14px;width:4px;
          border-radius:4px;background:linear-gradient(var(--blue),var(--violet),var(--cyan));}
        .decision-card span {font-size:.72rem;color:#5e69a1;font-weight:700;letter-spacing:.08em;}
        .decision-card strong {display:inline-block;margin-top:.28rem;color:var(--ink);font-weight:650;}
        .clause {padding:1rem 0;border-bottom:1px solid var(--border);line-height:1.6;}
        .focused-clause {padding:1.15rem 1.25rem;border:1px solid #bdd2fb;border-radius:14px;
          background:linear-gradient(110deg,#f2f7ff,#faf7ff);line-height:1.65;margin-top:1rem;}
        mark {background:#dff6ff;color:var(--ink);border-radius:4px;padding:1px 3px;}
        @media (max-width:640px) {
          [data-testid="stImage"] img {width:78%!important;transform:translateY(8px);}
          .hero-title {font-size:2rem;}
        }
        </style>""",
        unsafe_allow_html=True,
    )
    hero_copy, hero_visual = st.columns([1.6, .8], vertical_alignment="center")
    with hero_copy:
        st.markdown(
            """<div class="hero-copy">
              <div class="hero-kicker">Assistant contrats</div>
              <div class="hero-title">Une décision claire,<br>appuyée par les contrats.</div>
              <p class="hero-subtitle">Retrouvez les garanties applicables, vérifiez chaque réponse
              et accédez directement aux clauses utilisées.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with hero_visual:
        st.image(HERO_PATH, width="stretch")
    with st.sidebar:
        st.header("Repères d'utilisation")
        st.markdown("**Bien choisir la branche**")
        st.caption(
            "La branche correspond au type de contrat concerné : Automobile, Habitation, "
            "Santé, etc. Si vous connaissez le domaine de la question, sélectionnez-le dans "
            "le formulaire pour concentrer la recherche sur les contrats concernés. "
            "En cas de doute, conservez « Toutes les branches »."
        )
        st.divider()
        st.markdown("**Étendue de la recherche**")
        st.slider("Nombre de clauses consultées", 3, 8, key="top_k")
        st.caption(
            "L'assistant sélectionne ce nombre de clauses avant de répondre. "
            "Une valeur élevée est utile pour une question qui concerne plusieurs garanties, "
            "mais peut faire apparaître des clauses moins directement liées au dossier. "
            "La valeur recommandée est 5."
        )
        st.divider()
        st.markdown("**Aide à la décision**")
        st.caption(
            "L'assistant prépare une réponse à partir des contrats disponibles. "
            "Le gestionnaire reste responsable de la décision communiquée au client."
        )

    render_conversation_menu()
    active_view = st.segmented_control(
        "Navigation",
        [CHAT_VIEW, DOCUMENTS_VIEW],
        key="active_view",
        label_visibility="collapsed",
    )
    st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)
    if active_view == CHAT_VIEW:
        render_chat(get_api_key())
    else:
        render_documents()


if __name__ == "__main__":
    main()
