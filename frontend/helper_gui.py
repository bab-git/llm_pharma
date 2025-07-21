import gradio as gr
import pandas as pd
import os

class trials_gui( ):
    SENTINEL_NONE = "__NONE__"
    def __init__(self, graph, share=False):
        self.graph = graph
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 10
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        #self.sdisps = {} #global    
        self.demo = self.create_interface()

    def get_tab_notification(self, current_state):
        """Determine which tab to notify user about based on current state"""

        last_node = current_state.values.get("last_node", None)
        nnode = current_state.values.get("next", None)        
        if not last_node:
            return "Agent is working..."
        
        # Check for error messages first
        error_message = current_state.values.get("error_message", "")
        if error_message:
            return error_message
            
        policy_eligible = current_state.values.get("policy_eligible", None)
        trial_found = current_state.values.get("trial_found")
        trials = current_state.values.get("trials", [])
        # Map nodes to appropriate tab notifications
        node_to_tab = {
            'patient_collector': 'Check Patient Profile - Patient profile has been created',
            'policy_search': 'Check Policy Conflict - Relevant Clinical Policies have been retrieved', 
            'policy_evaluator': 'Check Policy Conflict - In case of policy conflict, your action is needed',
            'trial_search': 'Check Potential Trials - Potentially relevant clinical trials have been found',
            'grade_trials': 'Check Trials Scores - Calculated Trial Relevance scores are ready',            
            'profile_rewriter': 'Go to Profile Tab - Patient profile has been updated'
        }
        
        # Special case for policy issues
        if trial_found == True:
            return '🎉 Trial Scores - Perfectly Matched clinical trials have been found! 🎉'
        elif last_node == 'grade_trials':
            return """⚠️ Trials Scores - No matched trials found. Please review the relevance scores for more details.
Your options:            
   A - Continue with auto-generated profile rewriter --> Continue Evaluation,
   B - Manually modify the patient profile to better match trials.            
"""
        # elif last_node == 'policy_evaluator' and policy_eligible == False:
            # return 'Go to Policy Issue Tab - ATTENTION: Patient has policy conflicts that need review'
        # elif last_node == 'policy_evaluator' and policy_eligible == True:
            # return 'Agent continuing - Policy check passed, no action needed'
        elif last_node == 'trial_search' and trials == []:
            # if nnode == 'profile_rewriter':
                # return 'Profile - No potential trials found. \n Continue: Use profile rewriter or manually modify the patient profile.'
            if nnode == None:
                return 'Agent Tab - The pipeline couldn\'t find any potential/relevant trials. Try another patient.'
        elif last_node == 'profile_rewriter':
            return 'Go to Profile Tab - The patient profile has been rewritten by the agent to increase the chances of finding relevant trials. You can also manually modify the patient profile.'
            
        return node_to_tab.get(last_node, f'Agent is at: {last_node}')

    def run_agent(self, start, patient_prompt, stop_after):
        #global partial_message, thread_id,thread
        #global response, max_iterations, iterations, threads
        if start:
            self.iterations.append(0)
            # Get the current selected model from the state
            current_values = self.graph.get_state(self.thread)
            selected_model = current_values.values.get("selected_model", "llama-3.3-70b-versatile") if current_values.values else "llama-3.3-70b-versatile"
            
            config = {
                'patient_prompt': patient_prompt,
                "max_revisions": 3,
                "revision_number": 0,
                "trial_searches": 0,
                "max_trial_searches": 2,
                'last_node': "",
                'selected_model': selected_model}
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
        else:
            config = None
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            self.response = self.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_message += str(self.response)
            self.partial_message += f"\n {'='*40}\n\n"
            last_node, nnode, _, rev, acount = self.get_disp_state()

            # Default: don't update
            policies_update = gr.update()
            policy_issue_update = gr.update()
            trials_update = gr.update()

            # Update based on node
            if last_node == "policy_evaluator":
                policy_issue_update = self.get_last_policy_status()
            if last_node == "trial_search":
                trials_update = self.get_trials_summary_table()
            if last_node == "grade_trials":
                trials_update = self.get_trials_summary_table()

            # Only yield the 3 outputs expected by Gradio
            yield (
                self.partial_message,  # live
                policy_issue_update,  # policy_status
                trials_update         # trials_summary (must be a DataFrame or gr.update())
            )
            config = None
            if not nnode:
                return
            if last_node in stop_after:
                return
        return
    
    def get_disp_state(self,):
        current_state = self.graph.get_state(self.thread)
        last_node = current_state.values["last_node"]
        acount = current_state.values["revision_number"]
        rev = current_state.values["revision_number"]
        nnode = current_state.next
        #print  (last_node,nnode,self.thread_id,rev,acount)
        return last_node,nnode,self.thread_id,rev,acount
    
    def get_state(self,key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            last_node,nnode,self.thread_id,rev,astep = self.get_disp_state()
            new_label = f"last_node: {last_node}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return gr.update(label=new_label, value=current_values.values[key])
        else:
            return ""
    
    def get_state_value_only(self,key):
        """Get state value without changing the label - for components that should keep their original labels."""
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            return gr.update(value=current_values.values[key])
        else:
            return gr.update(value="")
    
    def get_patient_profile_with_formatting(self):
        """Get patient profile with preserved textbox formatting (multi-line support)."""
        current_values = self.graph.get_state(self.thread)
        if "patient_profile" in current_values.values:
            return gr.update(value=current_values.values["patient_profile"], lines=5)
        else:
            return gr.update(value="", lines=5)
    
    # def get_groq_models(self):
    #     """Get a list of Groq models with tool calling capabilities, sorted by performance."""
    #     return [
    #         # High Performance Models (Tool Calling + High Quality)
    #         ("llama-3.3-70b-versatile", "🦙 Llama 3.3 70B Versatile (Best)", "⭐⭐⭐⭐⭐"),
    #         ("llama-3.1-8b-versatile", "🦙 Llama 3.1 8B Versatile (Fast)", "⭐⭐⭐⭐"),
    #         ("llama-3.1-405b-reasoning", "🦙 Llama 3.1 405B Reasoning (Best Reasoning)", "⭐⭐⭐⭐⭐"),
    #         ("llama-3.1-70b-versatile", "🦙 Llama 3.1 70B Versatile (Balanced)", "⭐⭐⭐⭐"),
            
    #         # Good Performance Models
    #         ("llama-3.1-8b-instruct", "🦙 Llama 3.1 8B Instruct (Fast)", "⭐⭐⭐"),
    #         ("llama-3.1-70b-instruct", "🦙 Llama 3.1 70B Instruct (Good)", "⭐⭐⭐⭐"),
            
    #         # Alternative Models
    #         ("gemma2-9b-it", "💎 Gemma2 9B IT (Fast)", "⭐⭐⭐"),
    #         ("gemma2-27b-it", "💎 Gemma2 27B IT (Good)", "⭐⭐⭐⭐"),
    #         ("mixtral-8x7b-32768", "🎯 Mixtral 8x7B (Balanced)", "⭐⭐⭐⭐"),
            
    #         # Smaller/Faster Models
    #         ("llama-3.1-1b-instruct", "🦙 Llama 3.1 1B Instruct (Very Fast)", "⭐⭐"),
    #         ("llama-3.1-3b-instruct", "🦙 Llama 3.1 3B Instruct (Fast)", "⭐⭐⭐"),
    #     ]
    
    # def get_model_id_from_display_name(self, display_name):
    #     """Convert display name back to model ID."""
    #     models = self.get_groq_models()
    #     for model_id, disp_name, _ in models:
    #         if disp_name == display_name:
    #             return model_id
    #     return "llama-3.3-70b-versatile"  # Default fallback  
    
    def get_content(self, key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            documents = current_values.values[key]
            # print(policies)
            last_node,nnode,thread_id,rev,astep = self.get_disp_state()
            new_label = f"last_node: {last_node}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            if key == "policies":
                value = "\n\n".join(item.page_content for item in documents)
            elif key == "trials":
                value = "\n\n".join(f"{'-' * 40}\n\n#### The trial number {doc.metadata['nctid']}:\n{doc.page_content}" for doc in documents)
            return gr.update(label=new_label, value= value + "\n\n")
        else:
            return ""  

    def get_table(self, key = None):
        current_values = self.graph.get_state(self.thread)
        
        # Check if the state has been initialized with required keys
        if not current_values.values or "last_node" not in current_values.values:
            # Return placeholder data when state is not initialized
            if key == "trials_scores":
                return pd.DataFrame({
                    'nctid': ['No data available'],
                    'relevance_score': ['Start evaluation to see results'],
                    'explanation': ['Please run the agent evaluation first'],
                    'further_information': ['Click "Start Evaluation" in the Agent tab']
                })
            elif key == "trials":
                return pd.DataFrame({
                    'index': ['No data available'],
                    'nctid': ['Start evaluation to see results'],
                    'diseases': ['Please run the agent evaluation first'],
                    'Criteria': ['Click "Start Evaluation" in the Agent tab']
                })
            else:
                raise ValueError("key should be 'relevant_trials' or 'trials'")
        
        last_node = current_values.values["last_node"]
        df = pd.DataFrame(columns=['no data available'])
        
        if key == "trials_scores":
            if 'relevant_trials' in current_values.values:
                scores = current_values.values['relevant_trials']
                df = pd.DataFrame(scores)
                df = df.reindex(columns=['nctid', 'relevance_score','explanation', 'further_information'])            
            else:
                # Return placeholder when no trials have been scored yet
                df = pd.DataFrame({
                    'nctid': ['No trials scored yet'],
                    'relevance_score': ['Complete trial search first'],
                    'explanation': ['Trials need to be retrieved and graded'],
                    'further_information': ['Continue evaluation in the Agent tab']
                })
        elif key == "trials":            
            if 'trials' in current_values.values:                
                documents = current_values.values[key]                
                data = []
                for idx, doc in enumerate(documents):                    
                    page_content = doc.page_content
                    nctid = doc.metadata["nctid"]
                    diseases = doc.metadata["diseases"]
                    data.append({"index": idx, "nctid": nctid, "diseases": diseases , "Criteria": page_content})                
                df = pd.DataFrame(data)                
            else:
                # Return placeholder when no trials have been retrieved yet
                df = pd.DataFrame({
                    'index': ['No trials retrieved yet'],
                    'nctid': ['Complete policy evaluation first'],
                    'diseases': ['Trials will be searched after policy check'],
                    'Criteria': ['Continue evaluation in the Agent tab']
                })
        else:
            raise ValueError("key should be 'relevant_trials' or 'trials'")            
        return df

    def get_issue_policy(self,):
        current_values = self.graph.get_state(self.thread)
        value = ""
        new_label = "No policies checked yet"
        if "checked_policy" in current_values.values:
            checked_policy = current_values.values["checked_policy"]
            value += f"The last checked policy:\n{checked_policy.page_content}"
            lnode,nnode,thread_id,rev,astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
        
        if "policy_eligible" in current_values.values and current_values.values["policy_eligible"] == False:
            # checked_policy = current_values.values["checked_policy"]
            rejection_reason = current_values.values["rejection_reason"]
            value += f"""\nThe patient is rejected because of the following reason:
            {rejection_reason}
            """
            # return gr.update(label=new_label, value=value)
        else:
            value += "\n\n✅ **NO POLICY CONFLICTS FOUND YET**"
            # new_label = 
            
        return gr.update(label=new_label, value=value)

    def update_hist_pd(self,):
        #print("update_hist_pd")
        hist = []
        # curiously, this generator returns the latest first
        for state in self.graph.get_state_history(self.thread):
            if state.metadata['step'] < 1:
                continue
            thread_ts = state.config['configurable']['thread_ts']
            tid = state.config['configurable']['thread_id']
            revision_number = state.values['revision_number']
            last_node = state.values['last_node']
            rev = state.values['revision_number']
            nnode = state.next
            st = f"{tid}:{revision_number}:{last_node}:{nnode}:{rev}:{thread_ts}"
            hist.append(st)
        return gr.Dropdown(label="update_state from: thread:revision_number:last_node:next_node:rev:thread_ts", 
                           choices=hist, value=hist[0],interactive=True)
    
    def find_config(self,thread_ts):
        for state in self.graph.get_state_history(self.thread):
            config = state.config
            if config['configurable']['thread_ts'] == thread_ts:
                return config
        return(None)
            
    def copy_state(self,hist_str):
        ''' result of selecting an old state from the step pulldown. Note does not change thread. 
             This copies an old state to a new current state. 
        '''
        thread_ts = hist_str.split(":")[-1]
        #print(f"copy_state from {thread_ts}")
        config = self.find_config(thread_ts)
        #print(config)
        state = self.graph.get_state(config)
        self.graph.update_state(self.thread, state.values, as_node=state.values['last_node'])
        new_state = self.graph.get_state(self.thread)  #should now match
        new_thread_ts = new_state.config['configurable']['thread_ts']
        tid = new_state.config['configurable']['thread_id']
        revision_number = new_state.values['revision_number']
        last_node = new_state.values['last_node']
        rev = new_state.values['revision_number']
        nnode = new_state.next
        return last_node,nnode,new_thread_ts,rev,revision_number
    
    def update_thread_pd(self,):
        #print("update_thread_pd")
        return gr.Dropdown(label="choose thread", choices=threads, value=self.thread_id,interactive=True)
    
    def switch_thread(self,new_thread_id):
        #print(f"switch_thread{new_thread_id}")
        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id
        return 
    
    def modify_state(self, key, asnode,new_value):
        ''' gets the current state, modifes a single value in the state identified by key, and updates state with it.
        note that this will create a new 'current state' node. If you do this multiple times with different keys, it will create
        one for each update. Note also that it doesn't resume after the update
        key = the state value to be changed
        asnode = None, or the next node to transition to after the update
        new_value = The new value for the key
        Logic:
        - last node: patient_collector --> modify current state, next_node: policy_search
        - last node: policy_search --> modify current state, next_node: policy_evaluator
        - last node: policy_evaluator --> next: policy_evaluator
        '''
        if asnode == self.SENTINEL_NONE:
            asnode = None
        change_list = None
        current_states = list(self.graph.get_state_history(self.thread))
        last_node = current_states[0].values['last_node']
        
        if key == 'patient_profile':
            if last_node == 'patient_collector':
                print('patient_collector node')
                asnode = None
                i_state = 0
            elif last_node == 'policy_search':
                print('policy_search node')
                asnode = 'policy_search'
                i_state = 0
            elif last_node == 'policy_evaluator':
                asnode = 'policy_evaluator'
                i_state = 1
                change_list = [('policy_eligible', 'N/A')]
                # key_ext = 'policy_eligible'
                # val_ext = 'N/A'
            elif last_node == 'grade_trials':
                asnode = 'trial_search'
                i_state = 1                
            elif last_node == 'profile_rewriter':
                asnode = 'profile_rewriter'
                i_state = 0
            else:
                raise ValueError(f"unexpected last node {last_node}")
            current_values = current_states[i_state]
            current_values.values[key] = new_value
        elif key == 'policy_skip':
            current_values = current_states[1]
            change_list = [('policy_eligible', True), ('rejection_reason', 'N/A')]
            # current_values.values['policy_eligible'] = True
            current_values.values['unchecked_policies'].pop(0)    
            # current_values.values['rejection_reason'] = 'N/A'
            asnode = "policy_evaluator"
        elif key == 'policy_big_skip':
            current_values = current_states[1]
            change_list = [('policy_eligible', True), ('rejection_reason', 'N/A')]
            current_values.values['unchecked_policies'] = []
            asnode = "policy_evaluator"

        else:
            raise ValueError(f"unexpected key {key}")        
        
        if change_list is not None:
            for key_ext, val_ext in change_list:
                current_values.values[key_ext] = val_ext        

        # print(current_values.values)
        # print(current_values.next)
        # asnode = 'patient_collector'
        self.graph.update_state(self.thread, current_values.values, as_node=asnode)
        print('State updated')
        current_state = list(self.graph.get_state_history(self.thread))[0]
        print(current_state.next)
        return

    def get_trials_summary_table(self):
        """Get a summary table of trials with only nctid, diseases, and relevance columns."""
        current_values = self.graph.get_state(self.thread)
        
        # Check if the state has been initialized
        if not current_values.values or "last_node" not in current_values.values:
            return pd.DataFrame({
                'nctid': ['No data available'],
                'diseases': ['Start evaluation to see results'], 
                'relevance': ['Click "Start Evaluation" in the Agent tab']
            })
        
        # Check if we have relevant trials data (after grade_trials)
        if 'relevant_trials' in current_values.values and current_values.values['relevant_trials']:
            relevant_trials = current_values.values['relevant_trials']
            trials_data = []
            
            for trial in relevant_trials:
                nctid = trial.get('nctid', 'Unknown')
                # Get diseases from the original trials data if available
                diseases = 'Unknown'
                if 'trials' in current_values.values:
                    for orig_trial in current_values.values['trials']:
                        if orig_trial.metadata.get('nctid') == nctid:
                            diseases = orig_trial.metadata.get('diseases', 'Unknown')
                            break
                
                relevance = trial.get('relevance_score', 'Unknown')
                trials_data.append({
                    'nctid': nctid,
                    'diseases': diseases,
                    'relevance': relevance
                })
            
            return pd.DataFrame(trials_data)
        
        # If no relevant trials yet, check if we have basic trials data
        elif 'trials' in current_values.values and current_values.values['trials']:
            trials = current_values.values['trials']
            trials_data = []
            
            for trial in trials:
                nctid = trial.metadata.get('nctid', 'Unknown')
                diseases = trial.metadata.get('diseases', 'Unknown')
                trials_data.append({
                    'nctid': nctid,
                    'diseases': diseases,
                    'relevance': 'Not graded yet'
                })
            
            return pd.DataFrame(trials_data)
        
        else:
            return pd.DataFrame({
                'nctid': ['No trials found yet'],
                'diseases': ['Complete policy evaluation first'],
                'relevance': ['Trials will be searched after policy check']
            })

    def get_last_policy_status(self):
        """Get the status of the last checked policy."""
        current_values = self.graph.get_state(self.thread)
        
        if not current_values.values or "last_node" not in current_values.values:
            return gr.update(
                label="Policy Status",
                value="No policy evaluation started yet"
            )
        
        last_node = current_values.values.get("last_node", "")
        policy_status = "No policy checked yet"
        
        if "checked_policy" in current_values.values and current_values.values["checked_policy"]:
            checked_policy = current_values.values["checked_policy"]
            policy_eligible = current_values.values.get("policy_eligible", None)
            rejection_reason = current_values.values.get("rejection_reason", "")
            
            # Get policy title/header
            policy_content = checked_policy.page_content
            policy_header = policy_content.split('\n')[0] if policy_content else "Policy"
            
            if policy_eligible is True:
                status_icon = "✅"
                status_text = "PASSED"
                policy_status = f"{status_icon} Last Policy: {policy_header}\nStatus: {status_text}"
            elif policy_eligible is False:
                status_icon = "❌"
                status_text = "FAILED"
                policy_status = f"{status_icon} Last Policy: {policy_header}\nStatus: {status_text}"
                
                # Add rejection reason prominently
                if rejection_reason:
                    policy_status += f"\n\n🚨 **Rejection Reason:**\n{rejection_reason}"
            else:
                status_icon = "❓"
                status_text = "UNKNOWN"
                policy_status = f"{status_icon} Last Policy: {policy_header}\nStatus: {status_text}"
        
        # last_node, nnode, thread_id, rev, astep = self.get_disp_state()
        # new_label = f"Policy Status (last_node: {last_node}, rev: {rev})"
        
        return gr.update(value=policy_status)

    def get_stages_history(self):
        """Get the history of stages/nodes that have been executed."""
        stages_list = []
        
        try:
            # Get state history in reverse order (latest first)
            for state in self.graph.get_state_history(self.thread):
                if state.metadata.get('step', 0) < 1:  # Skip early states
                    continue
                
                last_node = state.values.get('last_node', 'unknown')
                revision_number = state.values.get('revision_number', 0)
                thread_ts = state.config['configurable'].get('thread_ts', 'unknown')
                
                # Create a stage entry
                stage_entry = f"Step {revision_number}: {last_node} (ts: {thread_ts[-8:]})"
                stages_list.append(stage_entry)
            
            # Reverse to show chronological order (oldest first)
            stages_list.reverse()
            
            if not stages_list:
                return gr.update(
                    label="Stages History",
                    value="No stages executed yet - Start evaluation to see progress"
                )
            
            stages_text = "\n".join(stages_list)
            last_node, nnode, thread_id, rev, astep = self.get_disp_state()
            new_label = f"Stages History (Thread: {thread_id}, Current: {last_node})"
            
            return gr.update(label=new_label, value=stages_text)
            
        except Exception as e:
            return gr.update(
                label="Stages History (Error)",
                value=f"Error retrieving stages: {str(e)}"
            )

    def get_current_policies(self):
        """Get the current policies from the agent state."""
        current_values = self.graph.get_state(self.thread)
        
        if not current_values.values or "policies" not in current_values.values:
            return gr.update(
                label="📜 Policies Related to the Patient",
                value="No policies loaded yet - Start evaluation to see policies"
            )
        
        # Check if we have policies in the state
        if current_values.values['policies']:
            policies = current_values.values['policies']
            policies_text = "\n\n".join(f"**Policy {i+1}:**\n{doc.page_content}" for i, doc in enumerate(policies))
            return gr.update(value=policies_text)
        else:
            return gr.update(
                label="📜 Policies Related to the Patient",
                value="No policies found in current state"
            )

    def create_interface(self):
        with gr.Blocks(
            theme=gr.themes.Default(spacing_size='sm',text_size="sm"),
            css="""
            .main-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
                padding: 20px !important;
            }
            .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
            }
            """
        ) as demo:
            
            # Add app description at the very top, before any tabs
            app_description = gr.Markdown(
                value="""## 🏥 Clinical Trial Eligibility Assistant

**Purpose:** This AI-powered application helps healthcare professionals evaluate patient eligibility for clinical trials by analyzing patient data, reviewing trial policies, and matching patients with appropriate studies.

**How it works:** The system uses a multi-stage evaluation process to assess patient compatibility with available clinical trials, ensuring both patient safety and trial requirements are met.""",
                visible=True
            )
            
            def updt_disp():
                ''' general update display on state change '''
                current_state = self.graph.get_state(self.thread)
                hist = []
                # curiously, this generator returns the latest first
                for state in self.graph.get_state_history(self.thread):
                    if state.metadata['step'] < 1:  #ignore early states
                        continue
                    s_thread_ts = state.config['configurable'].get('thread_ts', 'unknown')
                    s_tid = state.config['configurable']['thread_id']
                    s_count = state.values['revision_number']
                    s_last_node = state.values['last_node']
                    s_rev = state.values['revision_number']
                    s_nnode = state.next
                    st = f"{s_tid}:{s_count}:{s_last_node}:{s_nnode}:{s_rev}:{s_thread_ts}"
                    # print(st)
                    hist.append(st)
                if not current_state.metadata: #handle init call
                    return {
                        prompt_bx: "",
                        tab_notification: gr.update(value="Ready to start agent evaluation", visible=True),
                        last_node: "",
                        count_bx: "",
                        search_bx: "",
                        nnode_bx: "",
                        eligible_bx: "",
                        threadid_bx: self.thread_id,
                        thread_pd: gr.Dropdown(label="choose thread", choices=self.threads, value=self.thread_id, interactive=True),
                        step_pd: gr.Dropdown(label="update_state from: thread:revision_number:last_node:next_node:rev:thread_ts", 
                               choices=["N/A"], value="N/A", interactive=True),
                    }
                else:
                    # Get tab notification based on current state
                    # policy_eligible = current_state.values.get("policy_eligible")
                    # trial_found = current_state.values.get("trial_found")
                    notification = self.get_tab_notification(current_state)
                    
                    if current_state.values["policy_eligible"] == True:
                        eligible_bx_value = "✅ Yes"
                    elif current_state.values["policy_eligible"] == False:
                        eligible_bx_value = "❌ No"
                    else:
                        eligible_bx_value = "❓ Not determined"

                    return {
                        prompt_bx : current_state.values["patient_prompt"],
                        tab_notification: gr.update(value=notification, visible=True),
                        last_node : current_state.values["last_node"].replace("_", " ").title() if current_state.values["last_node"] else "",
                        count_bx : current_state.values["revision_number"],
                        search_bx : current_state.values["trial_searches"],
                        # revision_bx : current_state.values["revision_number"],
                        nnode_bx : current_state.next[0].replace("_", " ").title() if current_state.next and len(current_state.next) > 0 else ("END" if current_state.next is None else ""),
                        eligible_bx : eligible_bx_value,
                        threadid_bx : self.thread_id,
                        thread_pd : gr.Dropdown(label="choose thread", choices=self.threads, value=self.thread_id,interactive=True),
                        step_pd : gr.Dropdown(label="update_state from: thread:revision_number:last_node:next_node:rev:thread_ts", 
                               choices=hist, value=hist[0],interactive=True),
                    }
            def get_snapshots():
                new_label = f"thread_id: {self.thread_id}, Summary of snapshots"
                sstate = ""
                for state in self.graph.get_state_history(self.thread):
                    for key in ['plan', 'draft', 'critique']:
                        if key in state.values:
                            state.values[key] = state.values[key][:80] + "..."
                    if 'content' in state.values:
                        for i in range(len(state.values['content'])):
                            state.values['content'][i] = state.values['content'][i][:20] + '...'
                    if 'writes' in state.metadata:
                        state.metadata['writes'] = "not shown"
                    sstate += str(state) + "\n\n"
                return gr.update(label=new_label, value=sstate)

            def vary_btn(stat):
                #print(f"vary_btn{stat}")
                return(gr.update(variant=stat))
            
            with gr.Tab("Agent"):
                # Concise agent tab explanation with two columns
                with gr.Row():
                    with gr.Column(scale=1):
                        agent_info_left = gr.Markdown(
                            value="""## 🤖 Agent Control Center

**Start Here:**
- ➡️ **Enter Patient Query**
- ➡️ **Click 'Start Evaluation'**
- ➡️ **Monitor Progress via Notifications**

**Tips:**
- 💡 **Follow Notifications:** They guide you through each step
- 💡 **Use 'Manage Agent'** for advanced options and configurations
""",
                            visible=True
                        )
                    with gr.Column(scale=1):
                        agent_info_right = gr.Markdown(
                            value="""## Process Overview:

- 🔍 **Profile Creation:** Generate a patient profile
- ⚠️ **Policy Check:** Identify and resolve conflicting clinical policies
- 🔄 **Trial Matching:** Find potential clinical trial matches
- 🎯 **Trial Relevance:** Determine relevant trials for the patient
""",
                            visible=True
                        )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_bx = gr.Textbox(
                            label="Prompt about patient", 
                            value="Is patient_ID 56 eligible for any medical trial?",
                            lines=3,
                        )
                    with gr.Column(scale=1):
                        patient_id_dropdown = gr.Dropdown(
                            choices=[f"Patient {i}" for i in range(1, 100)],
                            label="Select Patient ID",
                            value="Patient 56",
                            interactive=True
                        )
                        # model_dropdown = gr.Dropdown(
                        #     choices=[display_name for _, display_name, _ in self.get_groq_models()],
                        #     label="🤖 Select AI Model",
                        #     value="🦙 Llama 3.3 70B Versatile (Best)",
                        #     interactive=True
                        # )
                    tab_notification = gr.Textbox(
                        value="Ready to start agent evaluation",
                        label="🔔 Your Notification Center 🔔",
                        lines=4,
                        interactive=False,
                        visible=True,
                        scale=3
                    )
                
                # Add processing indicator
                processing_status = gr.Markdown(
                    value="",
                    visible=False
                )
                
                with gr.Row():
                    gen_btn = gr.Button("Start Evaluation", scale=0,min_width=80, variant='primary')
                    cont_btn = gr.Button("Continue Evaluation", scale=0,min_width=80)
                    # Add debug mode switch
                    debug_mode = gr.Checkbox(label="🔧 Debug Mode", value=False, scale=0, min_width=120)
                
                with gr.Row():
                    last_node = gr.Textbox(label="Agent's last stop", min_width=150)
                    eligible_bx = gr.Textbox(label="Is Patient Eligible?", min_width=50)
                    nnode_bx = gr.Textbox(label="Agent's next step", min_width=150)
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=80, visible=False)
                    search_bx = gr.Textbox(label="trial_searches", scale=0, min_width=110, visible=False)
                    count_bx = gr.Textbox(label="revision_number", scale=0, min_width=110, visible=False)
                
                with gr.Accordion("Manage Agent", open=True):
                    checks = list(self.graph.nodes.keys())
                    checks.remove('__start__')
                    checks_values = checks.copy()
                    # Fix: remove() only takes one argument at a time and 'policy_searcj' is a typo.
                    # Should remove '__start__' and 'policy_search' if present.
                    # for node in ['__start__', 'policy_search', 'policy_evaluator', 'grade_trials', 'trial_search']:
                    #     if node in checks:
                    #         checks_values.remove(node)
                    stop_after = gr.CheckboxGroup(checks,label="Interrupt After State", value=checks_values, scale=0, min_width=400)
                    with gr.Row():
                        thread_pd = gr.Dropdown(choices=self.threads,interactive=True, label="select thread", min_width=120, scale=0, visible=False)
                        step_pd = gr.Dropdown(choices=['N/A'],interactive=True, label="select step", min_width=160, scale=1, visible=False)
                
                # NEW SECTION: Add the four requested components
                with gr.Accordion("", open=True):
                    # 1. Patient Profile Display
                    with gr.Row():
                        with gr.Column(scale=3):
                            profile_title = gr.Markdown(
                                value="## 📋 Current Patient Profile",
                                visible=True
                            )
                            profile_help = gr.Markdown(
                                value="""**✏️ You can directly change the text and press the 'Modify' button**, especially in cases of:
- ⚠️ **Conflicts with trial policies**
- 🔍 **When relevant trials cannot be found**
- 📝 **Helping patient to match specific trial**""",
                                visible=False
                            )
                        with gr.Column(scale=1):
                            modify_profile_btn = gr.Button("✏️ Modify Profile", min_width=120)
                    with gr.Row():
                        current_profile = gr.Textbox(
                            label="",
                            lines=5,
                            interactive=True,
                            placeholder="Patient profile will appear here after evaluation starts..."
                        )
                    
                    # 2. Policy Conflict Resolution
                    policy_title = gr.Markdown(
                        value="## ⚠️ Policy Conflict Resolution",
                        visible=True
                    )

                    with gr.Row():
                        with gr.Column(scale=3):
                            policy_conflict_info = gr.Markdown(
                                value="""**🚨 Policy Conflict Detected**

- **⏭️ Skip if this policy is not relevant**
- **🔧 Modify patient profile if needed**
- **⏭️⏭️ Skip all policy checks for the patient**""",
                                visible=False
                            )
                        with gr.Column(scale=1):
                            policy_skip_btn = gr.Button("⏭️ Skip Policy", min_width=120)
                            policy_big_skip_btn = gr.Button("⏭️ Skip All Policies", min_width=140)
                    
                    with gr.Row():
                        # Left column: Current Policies
                        with gr.Column(scale=1):
                            current_policies = gr.Textbox(
                                label="📜 Policies Related to the Patient", 
                                lines=15,
                                interactive=False,
                                placeholder="Current policies will appear here after policy search..."
                            )
                        
                        # Right column: Policy Issues
                        with gr.Column(scale=1):
                            policy_status = gr.Textbox(
                                label="⚠️ Policy Issues & Conflicts", 
                                lines=4,
                                interactive=False,
                                placeholder="Policy issues will appear here when conflicts are detected..."
                            )
                    
                    # 3. Trials Summary Table
                    with gr.Column():
                        trials_summary_heading = gr.Markdown(
                            value="""## 🎯 Trials Summary (NCT ID | Diseases | Relevance)

You can obtain more information about each trial's details and possible relevance reasons in the **Potential Trials** and **Trials Scores** tabs."""
                        )
                        trials_summary = gr.Dataframe(
                            label="",
                            headers=["nctid", "diseases", "relevance"],
                            interactive=False,
                            wrap=True
                        )
                    
                    # 4. Stages History Box
                    with gr.Row():
                        stages_history = gr.Textbox(
                            label="📈 Execution Stages History",
                            lines=6,
                            interactive=False,
                            placeholder="Stages execution history will appear here as the agent runs..."
                        )
                
                # Function to toggle debug fields visibility
                def toggle_debug_fields(debug_enabled):
                    return [
                        gr.update(visible=debug_enabled),  # threadid_bx
                        gr.update(visible=debug_enabled),  # search_bx
                        gr.update(visible=debug_enabled),  # count_bx
                        gr.update(visible=debug_enabled),  # thread_pd
                        gr.update(visible=debug_enabled),  # step_pd
                    ]
                
                # Function to show processing status
                def show_processing():
                    return gr.update(
                        value="🔄 **AGENT IS PROCESSING...** \n Please wait while the evaluation is in progress.",
                        visible=True
                    )
                
                # Function to hide processing status
                def hide_processing():
                    return gr.update(visible=False)
                
                # Function to refresh all status components
                def refresh_all_status(skip_policy_status_update=False):
                    """Refresh all the new status components."""
                    current_values = self.graph.get_state(self.thread)
                    last_node = current_values.values.get("last_node", "")
                    profile_ready = (current_values.values and 
                                   "patient_profile" in current_values.values and 
                                   current_values.values["patient_profile"] and 
                                   current_values.values["patient_profile"].strip())
                    policy_search_done = (current_values.values and 
                                        "last_node" in current_values.values and 
                                        current_values.values["last_node"] in ["policy_search", "policy_evaluator", "trial_search", "grade_trials", "profile_rewriter"])
                    policy_conflict_buttons_visible = (current_values.values and 
                                                     "last_node" in current_values.values and 
                                                     current_values.values["last_node"] == "policy_evaluator" and 
                                                     (current_values.next is None or len(current_values.next) == 0))
                    # Show "Skip All Policies" button after policy search is done
                    policy_big_skip_visible = policy_search_done
                    
                    return [
                        self.get_patient_profile_with_formatting(),  # current_profile - preserve multi-line formatting
                        self.get_current_policies(),        # current_policies - same as profile textbox behavior
                        self.get_last_policy_status() if not skip_policy_status_update else gr.update(),      # policy_status  
                        self.get_trials_summary_table(),    # trials_summary
                        self.get_stages_history(),          # stages_history
                        gr.update(visible=profile_ready),   # profile_help - show only when profile is ready
                        gr.update(visible=True),            # profile_title - always visible
                        gr.update(visible=policy_search_done),  # policy_conflict_info - show only when policy search is done
                        gr.update(visible=True),            # policy_title - always visible
                        gr.update(visible=policy_conflict_buttons_visible),  # policy_skip_btn - show only when policy conflict detected
                        gr.update(visible=policy_big_skip_visible)   # policy_big_skip_btn - show after policy search is done
                    ]
                
                # Add progress bar
                # progress_bar =  gr.Progress()
                
                with gr.Accordion("Live Agent Output - for debugging", open=False):
                    live = gr.Textbox(label="", lines=10, max_lines=25)
        
                # actions
                sdisps =[prompt_bx,tab_notification,last_node,eligible_bx, nnode_bx,threadid_bx,count_bx,step_pd,thread_pd, search_bx]
                
                # Add the new status components to the display list
                status_components = [current_profile, current_policies, policy_status, trials_summary, stages_history, profile_help, profile_title, policy_conflict_info, policy_title, policy_skip_btn, policy_big_skip_btn]
                
                # Add debug mode toggle functionality
                debug_mode.change(
                    fn=toggle_debug_fields,
                    inputs=[debug_mode],
                    outputs=[threadid_bx, search_bx, count_bx, thread_pd, step_pd]
                )
                

                
                # Wire up the modify profile button
                modify_profile_btn.click(
                    fn=self.modify_state, 
                    inputs=[gr.Number("patient_profile", visible=False),
                           gr.Number(self.SENTINEL_NONE, visible=False), 
                           current_profile],
                    outputs=None
                ).then(
                    fn=updt_disp, inputs=None, outputs=sdisps
                ).then(
                    fn=refresh_all_status, inputs=None, outputs=status_components
                )
                

                
                def skip_policy_and_notify():
                    """Skip the current policy and show confirmation message"""
                    # Skip the policy in the state
                    self.modify_state("policy_skip", self.SENTINEL_NONE, "")
                    # Return confirmation message
                    return gr.update(
                        label="Policy Skipped", 
                        value="The current policy is skipped for this patient.\n\nPlease continue evaluation of remaining policies in the Agent tab."
                    )

                def big_skip_policy_and_notify():
                    """Skip the whole policy check and show confirmation message"""
                    # Skip the policy in the state
                    self.modify_state("policy_big_skip", self.SENTINEL_NONE, "")
                    # Return confirmation message
                    return gr.update(
                        label="Policy Skipped", 
                        value="✅ The 'policy check phase' is completely skipped for this patient.\n\nPlease continue the next phase, Trial searches, via the Agent tab."
                    )
                
                def update_prompt_from_patient_id(patient_selection):
                    """Update the prompt text box when a patient ID is selected from dropdown"""
                    if patient_selection and patient_selection.startswith("Patient "):
                        patient_id = patient_selection.split(" ")[1]
                        return f"Is patient_ID {patient_id} eligible for any medical trial?"
                    return "Is patient_ID 56 eligible for any medical trial?"
                
                # def update_selected_model(model_display_name):
                #     """Update the selected model in the agent state"""
                #     model_id = self.get_model_id_from_display_name(model_display_name)
                #     # Update the state with the new model
                #     current_values = self.graph.get_state(self.thread)
                #     if current_values.values:
                #         current_values.values["selected_model"] = model_id
                #         self.graph.update_state(self.thread, current_values.values)
                #     return f"✅ Model updated to: {model_display_name}"
                
                policy_skip_btn.click(fn=skip_policy_and_notify, inputs=None, outputs=policy_status).then(
                                fn=updt_disp, inputs=None, outputs=sdisps).then(
                                fn=refresh_all_status, inputs=None, outputs=status_components)
                policy_big_skip_btn.click(fn=big_skip_policy_and_notify, inputs=None, outputs=policy_status).then(
                                fn=updt_disp, inputs=None, outputs=sdisps).then(
                                fn=lambda: refresh_all_status(skip_policy_status_update=True), inputs=None, outputs=status_components)
                
                # sdisps =[prompt_bx,last_node,eligible_bx, nnode_bx,threadid_bx,revision_bx,count_bx,step_pd,thread_pd]
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                                fn=updt_disp, inputs=None, outputs=sdisps).then(
                                fn=refresh_all_status, inputs=None, outputs=status_components)
                step_pd.input(self.copy_state,[step_pd],None).then(
                              fn=updt_disp, inputs=None, outputs=sdisps).then(
                              fn=refresh_all_status, inputs=None, outputs=status_components)
                gen_btn.click(vary_btn,gr.Number("secondary", visible=False), gen_btn).then(
                              vary_btn,gr.Number("primary", visible=False), cont_btn).then(
                              fn=show_processing, inputs=None, outputs=processing_status).then(
                              fn=self.run_agent, inputs=[gr.Number(True, visible=False),prompt_bx,stop_after], outputs=[live, trials_summary],show_progress=True).then(
                              fn=hide_processing, inputs=None, outputs=processing_status).then(
                              fn=updt_disp, inputs=None, outputs=sdisps).then(
                              fn=refresh_all_status, inputs=None, outputs=status_components)
                cont_btn.click(fn=show_processing, inputs=None, outputs=processing_status).then(
                               fn=self.run_agent, inputs=[gr.Number(False, visible=False),prompt_bx,stop_after], 
                               outputs=[live, policy_status, trials_summary]).then(
                               fn=hide_processing, inputs=None, outputs=processing_status).then(
                               fn=updt_disp, inputs=None, outputs=sdisps).then(
                               fn=refresh_all_status, inputs=None, outputs=status_components)
                
                # Wire up the patient ID dropdown to update the prompt
                patient_id_dropdown.change(
                    fn=update_prompt_from_patient_id,
                    inputs=[patient_id_dropdown],
                    outputs=[prompt_bx]
                )
                
                # Wire up the model dropdown to update the selected model
                # model_dropdown.change(
                #     fn=update_selected_model,
                #     inputs=[model_dropdown],
                #     outputs=[tab_notification]
                # )
        
                                              


                

            with gr.Tab("Potential Trials"):
                # Add informative text at the top of the Matched Trials tab
                matched_trials_info = gr.Markdown(
                    value="""## 🎯 Matched Clinical Trials

**📊 View trials** that match the patient's profile after policy evaluation.

**What you'll see:**
- **📋 Trial details** including study descriptions and requirements
- **✏️ Interactive table** - you can edit if needed
- **🔄 Refresh** to get updated trial matches

**Next step:**
- **➡️ Go to 'Trials Scores' tab** to see detailed scoring and ranking

**If no trials appear:**
- Check **'Profile'** tab to adjust patient information
- Review **'Policies'** tab for restrictive policies""",
                    visible=True
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    # modify_btn = gr.Button("Modify")
                # trials_bx = gr.Textbox(label="Retieved relevant trials based on patient's profile", lines=10, interactive=False)
                trials_bx = gr.Dataframe(label="Retieved relevant trials based on patient's profile", wrap=True, interactive=True)
                refresh_btn.click(fn=self.get_table, inputs=gr.Number("trials", visible=False), outputs=trials_bx)                
            
            with gr.Tab("Trials Scores"):
                # Add informative text at the top of the Trials Scores tab
                trials_scores_info = gr.Markdown(
                    value="""## 📊 Trial Eligibility Scores

**🏆 Final results** - Detailed scoring and ranking of matched trials.

**Understanding the score**🏆 Final results** - Detailed scoring and ranking of matched trials.

**Understanding the scores:**
- **✅ Relevant (Yes):** The patient's profile meets the trial's inclusion criteria and does not meet any exclusion criteria.
- **❌ Not Relevant (No):** The patient's profile meets any exclusion criteria or does not meet the trial's inclusion diseases.
- **🔄 Refresh** to get updated trial matches

**This is typically your final destination** in the evaluation process.

**If scores seem incorrect:**
- **🔙 Modify patient profile** in the 'Profile' tab
- **🔄 Re-run evaluation** from the 'Agent' tab""",
                    visible=True
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                # trials_scores_bx = gr.Textbox(label="Trials Scores based on patient's profile")
                trials_scores_bx = gr.Dataframe(label="Trials Scores based on patient's profile", wrap=True, interactive=False)
                refresh_btn.click(fn=self.get_table,  inputs=gr.Number("trials_scores", visible=False), outputs=trials_scores_bx)
            
            
            # with gr.Tab("StateSnapShots"):
            #     # Add informative text at the top of the StateSnapShots tab
            #     snapshots_info = gr.Markdown(
            #         value="""## 🔍 Agent State History

            # **🛠️ For debugging and advanced users** - View the complete agent execution history.

            # **What you'll find:**
            # - **📜 Step-by-step agent decisions** and state changes
            # - **🔧 Technical details** of the evaluation process
            # - **🕒 Historical snapshots** of each processing stage

            # **When to use:**
            # - **🐛 Troubleshooting** unexpected results
            # - **🔍 Understanding** how the agent reached its conclusions
            # - **⚙️ Advanced configuration** and debugging

            # **💡 Most users won't need this tab** - it's primarily for technical analysis.""",
            #         visible=True
            #     )
                
            #     with gr.Row():
            #         refresh_btn = gr.Button("Refresh")
            #     snapshots = gr.Textbox(label="State Snapshots Summaries")
            #     refresh_btn.click(fn=get_snapshots, inputs=None, outputs=snapshots)
        return demo

    def launch(self, share=None):
        if port := os.getenv("PORT1"):
            self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
        else:
            self.demo.launch(share=self.share)
