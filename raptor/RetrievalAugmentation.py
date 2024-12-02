import logging
import pickle

from cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from EmbeddingModels import BaseEmbeddingModel, LLaMAEmbeddingModel
from QAModels import BaseQAModel, LLaMAQAModel
from SummarizationModels import BaseSummarizationModel, LLaMASummarizationModel
from tree_builder import TreeBuilder, TreeBuilderConfig
from tree_retriever import TreeRetriever, TreeRetrieverConfig
from tree_structures import Node, Tree

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,
        qa_model=None,
        embedding_model=None,
        summarization_model=None,
        tree_builder_type="cluster",
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_context_embedding_model=None,
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=200,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model=None,
    ):
        # QA Model 설정
        if qa_model is None:
            qa_model = LLaMAQAModel()

        # Embedding model 설정
        if embedding_model is None:
            embedding_model = LLaMAEmbeddingModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
        
        if tb_embedding_models is None:
            tb_embedding_models = {"LLaMA_EMB": embedding_model}

        if tb_cluster_embedding_model is None or tb_cluster_embedding_model not in tb_embedding_models:
            tb_cluster_embedding_model = "LLaMA_EMB"

        # Summarization model 설정
        if summarization_model is None:
            summarization_model = LLaMASummarizationModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
        
        if tb_summarization_model is None:
            tb_summarization_model = summarization_model

        # Set TreeBuilderConfig
        tree_builder_class, tree_builder_config_class = supported_tree_builders[tree_builder_type]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
            )
        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Set TreeRetrieverConfig
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model or "LLaMA_EMB",  # 기본적으로 LLaMA 사용
                embedding_model=tr_embedding_model or embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError("tree_retriever_config must be an instance of TreeRetrieverConfig")

        # Assign the created configurations to the instance
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model
        self.summarization_model = summarization_model
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = f"""
        RetrievalAugmentationConfig:
            {self.tree_builder_config.log_config()}
            
            {self.tree_retriever_config.log_config()}
            
            QA Model: {self.qa_model}
            Tree Builder Type: {self.tree_builder_type}
        """
        return config_summary


class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree, retrieving information, and answering questions.
    """

    def __init__(self, config=None, tree=None):
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError("config must be an instance of RetrievalAugmentationConfig")

        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError("tree must be an instance of Tree, a path to a pickled Tree, or None")

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model
        self.summarization_model = config.summarization_model

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}")

    def add_to_existing(self, docs):
        """
        Adds new documents to the existing tree.
        """
        new_tree = self.tree_builder.build_from_text(text=docs)
        self.tree.hang_tree(new_tree)  # 새로운 트리를 기존 트리에 병합
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    # def add_documents(self, docs):
    #     """
    #     Adds documents to the tree and creates a TreeRetriever instance.
    #     """
    #     if self.tree is not None:
    #         user_input = input(
    #             "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
    #         )
    #         if user_input.lower() == "y":
    #             self.add_to_existing(docs)
    #             return

    #     self.tree = self.tree_builder.build_from_text(text=docs)
    #     self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
    def add_documents(self, docs):
        """
        Adds documents to the tree and creates a TreeRetriever instance.
        """
        # 기존 트리를 덮어쓰도록 구성
        if self.tree is not None:
            logging.info("Overwriting existing tree with new documents.")
        
        # 새 트리 생성 및 Retriever 인스턴스화
        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def retrieve(
        self,
        question,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = None,
        max_tokens: int = 12000,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Returns:
            str: The context from which the answer can be found.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        if self.retriever is None:
            raise ValueError("The TreeRetriever instance has not been initialized. Call 'add_documents' first.")
        
        # config에서 설정된 top_k를 사용하도록 수정
        if top_k is None:
            top_k = self.tree_retriever_config.top_k  # config에서 top_k 값을 가져옴

        return self.retriever.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, return_layer_information
        )

    def answer_question(
        self,
        question,
        top_k: int = None,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 12000,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            collapse_tree (bool): Whether to collapse the tree information or not.
            return_layer_information (bool): Whether to return layer information or not.

        Returns:
            str: The answer to the question.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        # config에서 설정된 top_k를 사용하도록 수정
        if top_k is None:
            top_k = self.tree_retriever_config.top_k  # config에서 top_k 값을 가져옴

        context, layer_information = self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True
        )

        answer = self.qa_model.answer_question(context, question)

        if return_layer_information:
            return answer, layer_information

        return answer

    def save(self, path):
        """
        Saves the current tree structure to the specified path.
        
        Args:
            path (str): The file path to save the tree.
        """
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")
