import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

def load_faq_documents(faq_path):
    with open(faq_path, encoding="utf-8") as f:
        content = f.read()
    pattern = r"^\s*Q\s*:(.*?)\r?\n\s*R\s*:(.*?)(?=\r?\n\s*Q\s*:|\Z)"
    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    docs = []
    for q, r in matches:
        q = q.strip().replace('\n', ' ')
        r = r.strip()
        if q and r:
            docs.append(Document(page_content=r, metadata={"question": q, "source": "FAQ", "type": "static"}))
    return docs

def trouver_formules(appareil, prix):
    tables_formules = {
        "pc": [
            (550, 1300, "Ifrah", 79, 15, 0, 0, 0),
            (1301, 2500, "Erteh", 99, 20, 0, 0, 0),
            (2501, 3600, "Thana", 129, 28, 0, 0, 0),
            (3601, 100000, "3ich", 163, 35, 0, 0, 0),
        ],
        "smartphone": [
            (250, 600, "Ifrah", 69, 15, 0, 0, 0),
            (601, 1200, "Erteh", 89, 20, 0, 0, 0),
            (1201, 2000, "Thana", 119, 28, 0, 0, 0),
            (2001, 3100, "Itmen", 179, 35, 0, 0, 0),
            (3101, 4500, "Tmata3", 239, 45, 0, 0, 0),
            (4501, 7899, "3ich", 299, 65, 0, 0, 0),
        ],
        "tablette": [
            (300, 900, "Ifrah", 69, 0, 0, 0, 0),
            (901, 1800, "Erteh", 89, 0, 0, 0, 0),
            (1801, 2700, "Thana", 129, 0, 0, 0, 0),
            (2701, 5000, "3ich", 179, 0, 0, 0, 0),
        ],
        "smartwatch": [
            (109, 299, "Erteh", 50, 15, 0, 0, 0),
            (300, 700, "Thana", 65, 25, 0, 0, 0),
            (701, 7000, "Itmen", 98, 35, 0, 0, 0),
        ],
        "tv": [
            (350, 750, "Ifrah", 73, 0, 0, 0, 0),
            (751, 1599, "Erteh", 96, 0, 0, 0, 0),
            (1600, 2600, "Thana", 123, 0, 0, 0, 0),
            (2601, 3990, "Itmen", 169, 0, 0, 0, 0),
            (3991, 6200, "3ich", 298, 0, 0, 0, 0),
        ],
        "equipements froids": [
            (500, 1399, "Ifrah", 60, 0, 0, 0, 0),
            (1400, 2699, "Erteh", 86, 0, 0, 0, 0),
            (2700, 5000, "Thana", 108, 0, 0, 0, 0),
        ],
        "equipements cuisine": [
            (250, 1000, "Ifrah", 45, 0, 0, 0, 0),
            (1001, 2000, "Thana", 75, 0, 0, 0, 0),
        ],
        "equipements ménagers": [
            (300, 1100, "Ifrah", 60, 0, 0, 0, 0),
            (1101, 2500, "Erteh", 78, 0, 0, 0, 0),
            (2501, 4000, "Thana", 119, 0, 0, 0, 0),
        ],
        "pack": [
            (750, 1599, "Erteh", 84, 0, 0, 0, 0),
            (1600, 3600, "Thana", 108, 0, 0, 0, 0),
        ],
        "macbook": [
            (3600, 5500, "Ifrah", 276, 0, 0, 0, 0),
            (5501, 7900, "Erteh", 345, 0, 0, 0, 0),
            (7901, 9000, "Thana", 399, 0, 0, 0, 0),
        ]
    }
    appareil = appareil.lower()
    if appareil not in tables_formules:
        return []
    formules = tables_formules[appareil]
    resultats = []
    for (pmin, pmax, nom, p1, vol1, p2, vol2, plafond) in formules:
        if pmin <= prix <= pmax:
            resultats.append({
                "formule": nom,
                "plafond": plafond,
                "prix_1an": str(p1),
                "option_vol_1an": str(vol1),
                "prix_2ans": str(p2),
                "option_vol_2ans": str(vol2)
            })
    if not resultats and formules:
        dernier = formules[-1]
        resultats.append({
            "formule": dernier[2],
            "plafond": dernier[7],
            "prix_1an": str(dernier[3]),
            "option_vol_1an": str(dernier[4]),
            "prix_2ans": str(dernier[5]),
            "option_vol_2ans": str(dernier[6]),
        })
    return resultats

class EnhancedRAGChain:
    def __init__(self, retriever, llm, system_prompt):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt

    def extract_entities(self, text):
        price, product = None, None
        synonyms = {
            'smartphone': ['smartphone', 'phone', 'portable', 'gsm', 'téléphone'],
            'pc': ['pc', 'ordinateur', 'computer', 'laptop', 'macbook', 'portable'],
            'tablette': ['tablette', 'ipad', 'tablet'],
            'smartwatch': ['smartwatch', 'montre connectée', 'montre'],
            'tv': ['tv', 'télévision'],
            'equipements froids': ['réfrigérateur', 'congélateur', 'climatiseur', 'réfrigérateur congélateur'],
            'equipements cuisine': ['four', 'micro-onde', 'hotte', 'table de cuisson', 'cuisinière', 'cuisuniére combinée'],
            'equipements ménagers': ['machine à laver', 'sèche linge', 'séche linge', 'séche-linge', 'congélateur'],
            'pack': ['pack'],
            'macbook': ['macbook']
        }
        text_lower = text.lower()
        price_match = re.search(r'(\d{1,7}(?:[.,]\d{1,2})?)\s*(dt|dinar|tnd|eur|usd|\$)?', text_lower)
        if price_match:
            try:
                price = float(price_match.group(1).replace(',', '.'))
                if not 10 <= price <= 1000000:
                    price = None
            except:
                price = None
        for canonical, terms in synonyms.items():
            if any(t in text_lower for t in terms):
                product = canonical
                break
        return price, product

    def contextualize_query(self, query, chat_history):
        price, product = self.extract_entities(query)
        if price is not None:
            st.session_state.collected_price = price
        if product is not None:
            st.session_state.collected_product = product
        final_price = st.session_state.collected_price
        final_product = st.session_state.collected_product
        if final_product:
            if final_price is not None:
                formules = trouver_formules(final_product, final_price)
                if formules:
                    texte = f"Pour votre {final_product} d'une valeur de {final_price} DT, voici les formules d'assurance compatibles :\n\n"
                    for i, f in enumerate(formules, 1):
                        try:
                            prix1 = int(f['prix_1an']) if f['prix_1an'].isdigit() else 0
                            opt1 = int(f['option_vol_1an']) if f['option_vol_1an'].isdigit() else 0
                            prix2 = int(f['prix_2ans']) if f['prix_2ans'].isdigit() else 0
                            opt2 = int(f['option_vol_2ans']) if f['option_vol_2ans'].isdigit() else 0
                        except:
                            prix1 = opt1 = prix2 = opt2 = 0
                        texte += f"{i}. Formule {f['formule']}\n"
                        texte += "   Assurance 1 an :\n"
                        if prix1 > 0:
                            texte += f"     - Prix de base : {f['prix_1an']} DT\n"
                        if opt1 > 0:
                            texte += f"     - Avec Option Vol : {f['prix_1an']} DT + {f['option_vol_1an']} DT = {prix1 + opt1} DT\n"
                        texte += "   Assurance 2 ans :\n"
                        if prix2 > 0:
                            texte += f"     - Prix de base : {f['prix_2ans']} DT\n"
                        if opt2 > 0:
                            texte += f"     - Avec Option Vol : {f['prix_2ans']} DT + {f['option_vol_2ans']} DT = {prix2 + opt2} DT\n\n"
                    return texte
                else:
                    return "❌ Aucun forfait trouvé pour ce prix."
            else:
                return f"Pour assurer un {final_product}, veuillez fournir son prix."
        else:
            return "Quel type d'appareil voulez-vous assurer ? (ex: pc, smartphone...)"

    def format_enhanced_context(self, retrieved_docs, query):
        return "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    def invoke(self, inputs):
        query = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        current_price, current_product = self.extract_entities(query)
        if current_price is not None:
            st.session_state.collected_price = current_price
        if current_product is not None:
            st.session_state.collected_product = current_product

        final_price = st.session_state.collected_price
        final_product = st.session_state.collected_product

        question_lower = query.lower()
        has_price_word = bool(re.search(r'\b(prix|tarif|coût|combien|valeur)\b', question_lower))
        has_device_word = any(device in question_lower for device in [
            "pc", "ordinateur", "smartphone", "phone", "portable", "tablette", "ipad",
            "smartwatch", "montre", "tv", "télévision", "réfrigérateur", "four",
            "machine à laver", "pack", "macbook"
        ])

        if final_price is not None and final_product is not None and (has_price_word or has_device_word):
            texte_formules = self.contextualize_query(query, chat_history)
            return {"output": texte_formules}
        else:
            retrieved_docs = self.retriever.get_relevant_documents(query)
            if not retrieved_docs:
                all_docs = list(self.retriever.vectorstore.docstore._dict.values())
                for doc in all_docs:
                    if any(w in doc.page_content.lower() for w in query.lower().split() if len(w) > 2):
                        retrieved_docs.append(doc)
                        if len(retrieved_docs) >= 5:
                            break
            context = self.format_enhanced_context(retrieved_docs, query)
            messages = [SystemMessage(content=self.system_prompt.format(context=context))]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=query))
            response = self.llm.invoke(messages)
            result = {"output": response.content}
            if st.session_state.get("show_sources", False):
                unique_sources = {(doc.page_content, doc.metadata.get("url", doc.metadata.get("question"))): doc for doc in retrieved_docs}
                result["sources"] = list(unique_sources.values())
            return result

def main():
    st.set_page_config(page_title="Chatbot FAQ Assurance", page_icon="🤖", layout="wide")
    st.title("🤖 Assistant FAQ - Services d'assurance")
    st.markdown("Posez-moi vos questions sur nos services d'assurance ou indiquez votre type d'appareil et son prix.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "enhanced_rag_chain" not in st.session_state:
        st.session_state.enhanced_rag_chain = None
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False
    if "collected_price" not in st.session_state:
        st.session_state.collected_price = None
    if "collected_product" not in st.session_state:
        st.session_state.collected_product = None
    if "user_input_box" not in st.session_state:
        st.session_state.user_input_box = ""

    if not st.session_state.vector_store_initialized:
        faq_path = "FAQ.txt"
        if not os.path.exists(faq_path):
            st.error(f"Le fichier {faq_path} est introuvable. Merci de le placer au bon endroit.")
            st.stop()

        faq_documents = load_faq_documents(faq_path)
        st.session_state.faq_documents = faq_documents

       embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


        vectorstore = FAISS.from_documents(faq_documents, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        system_prompt = """
Vous êtes un assistant spécialisé dans l'assurance Garanty. Répondez de façon claire et concise 
en vous basant uniquement sur les informations fournies ci-dessous. Pour les questions sur les tarifs,
utilisez les tables tarifaires internes.

{context}
"""

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=GEMINI_API_KEY)
        st.session_state.enhanced_rag_chain = EnhancedRAGChain(retriever, llm, system_prompt)
        st.session_state.vector_store_initialized = True

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Votre question :", key="user_input_box", value=st.session_state.user_input_box or "")
        submitted = st.form_submit_button("Envoyer")

    if submitted and user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Réflexion en cours..."):
            try:
                response_dict = st.session_state.enhanced_rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.messages[:-1]
                })
                st.session_state.messages.append(AIMessage(content=response_dict["output"]))
                if st.session_state.get("show_sources", False) and "sources" in response_dict:
                    st.sidebar.subheader("Sources :")
                    for doc in response_dict["sources"]:
                        source = doc.metadata.get("source", "Inconnue")
                        if source == "FAQ":
                            st.sidebar.markdown(f"- **FAQ :** {doc.metadata.get('question')}")
                        elif source == "Web":
                            st.sidebar.markdown(f"- **Web :** {doc.metadata.get('url', '#')}")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")
                st.session_state.messages.append(AIMessage(content="Erreur de traitement."))

    for msg in st.session_state.messages:
        role = "Utilisateur" if isinstance(msg, HumanMessage) else "Assistant"
        st.markdown(f"**{role} :** {msg.content}")

    if not st.session_state.messages:
        st.info("Posez-moi une question pour commencer !")

    st.markdown("---\n🔗 Propulsé par Gemini, FAISS, LangChain & Streamlit")

if __name__ == "__main__":
    main()
