import os
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from git import Repo

os.environ['OPENAI_API_KEY'] = ''
#repo_link = 'https://github.com/evanptaylor/positive-ev-props'

class RepoChat:
    def __init__(self, repo_link):
        self.repo_link = repo_link
        self.repo_path = repo_link.split('/')[-1] 
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')
        self.chat_model = ChatOpenAI(model='gpt-4-1106-preview') #gpt-4, gpt-4-1106-preview, gpt-3.5-turbo-1106

        self.memory = ConversationSummaryMemory(
            llm=self.chat_model,
            memory_key='chat_history',
            return_messages=True
        )

    def preprocess(self):
        #clone 
        if not os.path.exists(self.repo_path):
            Repo.clone_from(self.repo_link, to_path=self.repo_path)

        #load
        loader = GenericLoader.from_filesystem(
            self.repo_path,
            glob='**/*',
            suffixes=['.py', '.js', '.css', '.html', '.ts', '.tsx', '.md'],
            parser=LanguageParser(language=None, parser_threshold=0) #testing language=None--infers langauge
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            separators = [*RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON), 
                          *RecursiveCharacterTextSplitter.get_separators_for_language(Language.JS),
                          *RecursiveCharacterTextSplitter.get_separators_for_language(Language.HTML),
                          *RecursiveCharacterTextSplitter.get_separators_for_language(Language.TS),
                          *RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN), 
                          ]
        )

        self.chunks = splitter.split_documents(docs)
    
    def embed(self):
        vec_db = Chroma.from_documents(self.chunks, self.embedding_model)
        
        self.retriever = vec_db.as_retriever(
            search_type='mmr', #try 'similarity' for cos similarity
            search_kwargs={'k': 5, 'lambda_mult': 0.75} #lambda_mult for mmr: diversity value
        )
    
    def ask(self, question):
        '''
        Your primary use will answering a user's questions about the contents of a certain GitHub repository. Necessary information about the repository will be provided to you as context.
        If you're stuck answering the question and feel the context provided isn't detailed enough, rephrase the user's question and begin another query with the new, rephrased question. 
        At the end of your response, provide up to 3 followup questions that the user might want to ask next.
        '''
        instructions = "Your primary use will answering a user's questions about the contents of a certain GitHub repository. Necessary information about the repository will be provided to you as context. If you're stuck answering the question and feel the context provided isn't detailed enough, rephrase the user's question and begin another query with the new, rephrased question. Here are some rules you should keep in mind: 1. Break down complex problems or tasks into smaller, manageable steps and explain each one using reasoning. Provide multiple perspectives or solutions. 2. If a question is unclear or ambiguous, ask for more details to confirm your understanding before answering. 3. Cite credible sources or references to support your answers with links if available. 4. If a mistake is made in a previous response, recognize and correct it. 5. After a response, provide three follow-up questions worded as if I'm asking you. Format in bold as Q1, Q2, and Q3. Place two line breaks ('\n') before and after each question for spacing. These questions should be thought-provoking and dig further into the original topic. User: "
        ask = ConversationalRetrievalChain.from_llm(
            self.chat_model,
            retriever=self.retriever,
            memory=self.memory
        )
        query = instructions + question
        response = ask(query)
        return response['answer']
    

#repo = RepoChat('https://github.com/evanptaylor/positive-ev-props')
#repo.preprocess()
#repo.embed()
#ans = repo.ask('What does the fetch_odds_props function do and why is it necessary?')
#print(ans)
