"""
Trial Service Module

Owns everything about matching a patient profile to trials and scoring relevance.
No leakage of DB or policy details.
"""

from typing import Optional

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from .database_manager import DatabaseManager
from .llm_manager import LLMManager
from .State import AgentState


def safe_invoke(retriever, question, retries=2):
    for i in range(retries):
        try:
            return retriever.invoke(question)
        except ValueError:
            print(f"⚠️ JSON parse failed (attempt {i+1}), retrying…")
    raise RuntimeError("Failed to parse JSON after retries")


# --- Structured Output Schemas ---
class grade(BaseModel):
    """The result of the trial's relevance check as relevance score and explanation."""

    relevance_score: str = Field(description="Relevance score: 'Yes' or 'No'")
    explanation: str = Field(description="Reasons to the given relevance score.")
    further_information: Optional[str] = Field(
        default="Not applicable",
        description="Additional information needed from patient's medical history",
    )

    # class Config:
    #     json_schema_extra = {
    #         "example": {
    #             "relevance_score": "Yes",
    #             "explanation": "The patient has the target disease condition for this trial.",
    #             "further_information": "Need to verify patient's current treatment status.",
    #         }
    #     }


class GradeHallucinations(BaseModel):
    """Binary score and explanation for whether the LLM's generated answer is grounded in / supported by the facts in the patient's medical profile."""

    binary_score: str = Field(
        description="Answer is grounded in the patient's medical profile, 'yes' or 'no'"
    )
    reason: str = Field(
        description="Reasons to the given relevance score."
    )


# --- Helper to get default LLMManagers ---
def get_default_llm_managers():
    return LLMManager.get_default_managers()


# --- Trial Search Node ---
def trial_search_node(state: AgentState, configs: Optional[DictConfig] = None) -> dict:
    try:
        if configs is not None:
            llm_manager, llm_manager_tool = get_default_llm_managers()
            # collector_config = PatientCollectorConfig.from_config(configs)
            db_manager = DatabaseManager(configs=configs)
        else:
            llm_manager, llm_manager_tool = get_default_llm_managers()
            # collector_config = PatientCollectorConfig(
            #     llm_manager=llm_manager, llm_manager_tool=llm_manager_tool
            # )
            db_manager = DatabaseManager()
        patient_profile = state.get("patient_profile", "")
        if not patient_profile:
            print("⚠️ No patient profile available for trial search")
            return {
                "last_node": "trial_search",
                "trials": [],
                "trial_searches": state.get("trial_searches", 0) + 1,
                "policy_eligible": state.get("policy_eligible", False),
            }
        trial_vectorstore = db_manager.create_trial_vectorstore()
        print(
            f"Number of trials in the vector store: {trial_vectorstore._collection.count()}"
        )
        metadata_field_info = [
            AttributeInfo(
                name="disease_category",
                description="Defines the disease group of patients related to this trial. One of ['cancer', 'leukemia', 'mental_health']",
                type="string",
            ),
            AttributeInfo(
                name="drugs",
                description="List of drug names used in the trial",
                type="str",
            ),
        ]
        document_content_description = (
            "The list of patient conditions to include or exclude them from the trial"
        )
        print(f"patient_profile: {patient_profile}")
        question = f"""
        Which trials are relevant to the patient with the following medical history?\n
        patient_profile: {patient_profile}
        """

        def run_trial_retrieval():
            current_model = llm_manager.current
            retriever_trial_sq = SelfQueryRetriever.from_llm(
                current_model,
                trial_vectorstore,
                document_content_description,
                metadata_field_info,
            )
            return safe_invoke(retriever_trial_sq, question)

        docs_retrieved = llm_manager.invoke_with_fallback(
            run_trial_retrieval, reset=True
        )
        print(f"✅ Retrieved {len(docs_retrieved)} relevant trials")
        trial_searches = state.get("trial_searches", 0) + 1
        return {
            "last_node": "trial_search",
            "trials": docs_retrieved,
            "trial_searches": trial_searches,
            "policy_eligible": state.get("policy_eligible", False),
        }
    except Exception as e:
        print(f"❌ Error in trial search: {e}")
        return {
            "last_node": "trial_search",
            "trials": [],
            "trial_searches": state.get("trial_searches", 0) + 1,
            "policy_eligible": state.get("policy_eligible", False),
            "error_message": str(e) if e else "",
        }


# --- Grade Trials Node ---
def grade_trials_node(state: AgentState) -> dict:
    try:
        print("----- CHECKING THE TRIALS RELEVANCE TO PATIENT PROFILE ----- ")
        trial_found = False
        trials = state.get("trials", [])
        patient_profile = state.get("patient_profile", "")
        if not trials:
            print("⚠️ No trials available for grading")
            return {
                "last_node": "grade_trials",
                "relevant_trials": [],
                "policy_eligible": state.get("policy_eligible", False),
            }
        if not patient_profile:
            print("⚠️ No patient profile available for trial grading")
            return {
                "last_node": "grade_trials",
                "relevant_trials": [],
                "policy_eligible": state.get("policy_eligible", False),
            }
        llm_manager, llm_manager_tool = get_default_llm_managers()
        # config = PatientCollectorConfig(
        #     llm_manager=llm_manager, llm_manager_tool=llm_manager_tool
        # )
        relevant_trials = []
        for trial in trials:
            doc_txt = trial.page_content
            trial_diseases = trial.metadata["diseases"]
            nctid = trial.metadata["nctid"]
            print(f"---GRADER: TRIAL {nctid}: ---")

            def run_trial_score():
                current_model = llm_manager_tool.current
                prompt_grader = PromptTemplate(
                    template="""
                    You are a Principal Investigator (PI) for evaluating patients for clinical trials.\n
                    Your task is to evaluate the relevance of a clinical trial to the given patient's medical profile. \n
                    The clinical trial is related to these diseases: {trial_diseases} \n
                    Here are the inclusion and exclusion criteria of the trial: \n\n {document} \n\n
                    ===============
                    Use the following steps to determine relevance and provide the necessary fields in your response: \n
                    1- If the patient's profile meets any exclusion criteria, then the trial is not relevant --> relevance_score = 'No'. \n
                    2- If the patient has or had the trial's inclusion diseases, then it is relevant --> relevance_score = 'Yes'.\n
                    3- If the patient did not have the trial's inclusion diseases, then it is not relevant --> relevance_score = 'No'.\n
                    Example 1:
            The patient has Arthritis and the trial is related to pancreatic cancer. --> relevance_score = 'No' \n
                    Example 2:
            The patient has pancreatic cancer and the trial is also related to carcinoma pancreatic cancer. --> relevance_score = 'Yes' \n
                    Example 3:
            The patient has pancreatic cancer and the trial is related to breast cancer or ovarian cancer. --> relevance_score = 'No'. \n
                    Bring your justification in the explanation. \n
                    Mention further information that is needed from the patient's medical history related to the trial's criteria \n
                    ===============
                    Here is the patient's medical profile: {patient_profile} \n\n

                    Respond with:
                    - relevance_score: "Yes" or "No"
                    - explanation: Your reasoning
                    - further_information: What additional info is needed
                    """,
                    input_variables=["document", "patient_profile", "trial_diseases"],
                )
                try:
                    llm_with_tool = current_model.with_structured_output(grade)
                    retrieval_grader = prompt_grader | llm_with_tool
                    result = retrieval_grader.invoke(
                        {
                            "patient_profile": patient_profile,
                            "document": doc_txt,
                            "trial_diseases": trial_diseases,
                        }
                    )
                    # print(f"Grade result: {result}")
                    return result
                except Exception as e:
                    print(f"Structured output failed, using fallback: {e}")
                    text_response = (
                        prompt_grader | current_model | StrOutputParser()
                    ).invoke(
                        {
                            "patient_profile": patient_profile,
                            "document": doc_txt,
                            "trial_diseases": trial_diseases,
                        }
                    )
                    relevance = "No"  # Default to No for safety
                    if (
                        "yes" in text_response.lower()
                        and "relevance" in text_response.lower()
                    ):
                        relevance = "Yes"
                    return grade(
                        relevance_score=relevance,
                        explanation=text_response[:500],
                        further_information="Additional patient history review needed",
                    )

            trial_score = llm_manager_tool.invoke_with_fallback(
                run_trial_score, reset=False
            )
            relevance_score = trial_score.relevance_score
            trial_score_dic = dict(trial_score)
            trial_score_dic["nctid"] = nctid
            if relevance_score.lower() == "yes":
                explanation = trial_score.explanation

                def run_hallucination():
                    current_model = llm_manager_tool.current
                    prompt_hallucination = PromptTemplate(
                        template="""
                        You are a grader assessing whether an LLM generation is grounded in / supported by the facts in the patient's medical profile. \n
                        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the facts in the patient's medical profile.
                        ===============
                        Here is the patient's medical profile: {patient_profile} \n\n
                        ===============
                        Here is the LLM generated answer: {explanation} \n\n
                                                
                        Respond with:
                        - binary_score: "yes" or "no"
                        - reason: Your reasoning
                        """,
                        input_variables=["patient_profile", "explanation"],
                    )
                    
                    system = """You are a grader assessing whether an LLM generation is grounded in / supported by the facts in the patient's medical profile. \n
                         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the facts in the patient's medical profile."""
                    hallucination_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system),
                            (
                                "human",
                                "Patient's medical profile: \n\n {patient_profile} \n\n LLM generated answer: {explanation}",
                            ),
                        ]
                    )

                    llm_with_tool_hallucination = current_model.with_structured_output(
                        GradeHallucinations
                    )
                    # hallucination_grader = hallucination_prompt | llm_with_tool_hallucination
                    hallucination_grader = prompt_hallucination | llm_with_tool_hallucination
                    
                    result = hallucination_grader.invoke(
                        {
                            "patient_profile": patient_profile,
                            "explanation": explanation,
                        }
                    )
                    return result

                factual_score = llm_manager_tool.invoke_with_fallback(
                    run_hallucination, reset=False
                )
                factual_score_grade = factual_score.binary_score
                if factual_score_grade == "no":
                    print(
                        "--- HALLUCINATION: MODEL'S EXPLANATION IS NOT GROUNDED IN PATIENT PROFILE --> REJECTED---"
                    )
                    trial_score_dic["relevance_score"] = "no"
                    trial_score_dic["explanation"] = "Agent's Hallucination"
            if (
                relevance_score.lower() == "yes"
                and trial_score_dic.get("relevance_score", "").lower() == "yes"
            ):
                print("---TRIAL RELEVANT---")
                trial_found = True
            else:
                print("--- TRIAL NOT RELEVANT---")
            relevant_trials.append(trial_score_dic)
        return {
            "last_node": "grade_trials",
            "relevant_trials": relevant_trials,
            "policy_eligible": state.get("policy_eligible", False),
            "trial_found": trial_found,
        }
    except Exception as e:
        print(f"❌ Error in trial grading: {e}")
        return {
            "last_node": "grade_trials",
            "relevant_trials": [],
            "policy_eligible": state.get("policy_eligible", False),
            "error_message": str(e) if e else "",
        }
