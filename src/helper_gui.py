import gradio as gr
import pandas as pd

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

    def get_tab_notification(self, last_node, policy_eligible=None):
        """Determine which tab to notify user about based on current state"""
        if not last_node:
            return "Agent is starting..."
        
        # Map nodes to appropriate tab notifications
        node_to_tab = {
            'patient_collector': 'Profile - Patient profile has been created',
            'policy_search': 'Policies - Relevant policies have been retrieved', 
            'policy_evaluator': 'Policy Issue - Policy evaluation completed',
            'trial_search': 'Matched Trials - Clinical trials have been found',
            'grade_trials': 'Trials Scores - Trial relevance scores are ready',
            'profile_rewriter': 'Profile - Patient profile has been updated'
        }
        
        # Special case for policy issues
        if last_node == 'policy_evaluator' and policy_eligible == False:
            return 'Policy Issue - ATTENTION: Patient has policy conflicts that need review'
        elif last_node == 'policy_evaluator' and policy_eligible == True:
            return 'Agent continuing - Policy check passed, no action needed'
            
        return node_to_tab.get(last_node, f'Agent is at: {last_node}')

    def run_agent(self, start,patient_prompt,stop_after):
        #global partial_message, thread_id,thread
        #global response, max_iterations, iterations, threads
        if start:
            self.iterations.append(0)
            config = {
                'patient_prompt': patient_prompt,
                "max_revisions": 10,
                "revision_number": 0,
                "trial_searches": 0,
                "max_trial_searches": 3,
                'last_node': ""}
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
        else:
            config = None
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            self.response = self.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_message += str(self.response)
            self.partial_message += f"\n {40*'========='}\n\n"
            ## fix
            last_node,nnode,_,rev,acount = self.get_disp_state()
            yield self.partial_message,last_node,nnode,self.thread_id,rev,acount,
            config = None #need
            #print(f"run_agent:{last_node}")
            if not nnode:  
                #print("Hit the end")
                return
            if last_node in stop_after:
                #print(f"stopping due to stop_after {last_node}")
                return
            else:
                #print(f"Not stopping on last_node {last_node}")
                pass
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
        last_node = current_values.values["last_node"]
        df = pd.DataFrame(columns=['no data available'])
        if key == "trials_scores":
            if 'relevant_trials' in current_values.values:
                scores = current_values.values['relevant_trials']
                df = pd.DataFrame(scores)
                df = df.reindex(columns=['nctid', 'relevance_score','explanation', 'further_information'])            
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
\nYou can correct the patient's medical profile if required.            
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
    
    def modify_state(self,key,asnode,new_value):
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


    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Default(spacing_size='sm',text_size="sm")) as demo:
            
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
                    policy_eligible = current_state.values.get("policy_eligible")
                    notification = self.get_tab_notification(current_state.values["last_node"], policy_eligible)
                    
                    if current_state.values["policy_eligible"] == True:
                        eligible_bx_value = "✅ Yes"
                    elif current_state.values["policy_eligible"] == False:
                        eligible_bx_value = "❌ No"
                    else:
                        eligible_bx_value = "❓ Not determined"

                    return {
                        prompt_bx : current_state.values["patient_prompt"],
                        tab_notification: gr.update(value=notification, visible=True),
                        last_node : current_state.values["last_node"],
                        count_bx : current_state.values["revision_number"],
                        search_bx : current_state.values["trial_searches"],
                        # revision_bx : current_state.values["revision_number"],
                        nnode_bx : current_state.next,
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
                # Concise agent tab explanation
                agent_info = gr.Markdown(
                    value="""## 🤖 Agent Control Center

**🚀 Start here:** Enter patient query → Click 'Start Evaluation' → Monitor progress via notifications

**Advanced Options:** 
- **⏸️ 'Continue Evaluation'** to resume
- **⚙️ 'Manage Agent'** for interrupt configuration
- **🔄 Thread management** for session switching

**💡 Tip:** Follow notification guidance for next steps!""",
                    visible=True
                )
                
                with gr.Row():
                    prompt_bx = gr.Textbox(label="Patient Prompt", value="Is patient_ID 56 eligible for any medical trial?")
                    gen_btn = gr.Button("Start Evaluation", scale=0,min_width=80, variant='primary')
                    cont_btn = gr.Button("Continue Evaluation", scale=0,min_width=80)
                    # Add debug mode switch
                    debug_mode = gr.Checkbox(label="🔧 Debug Mode", value=False, scale=0, min_width=120)
                
                # Add notification box below patient prompt
                tab_notification = gr.Textbox(
                    value="Ready to start agent evaluation",
                    label="🔔 Check this tab",
                    interactive=False,
                    visible=True,
                )
                
                # Add processing indicator
                processing_status = gr.Markdown(
                    value="",
                    visible=False
                )
                
                with gr.Row():
                    last_node = gr.Textbox(label="last node", min_width=150)
                    eligible_bx = gr.Textbox(label="Is Patient Eligible?", min_width=50)
                    nnode_bx = gr.Textbox(label="next node", min_width=150)
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=80, visible=False)
                    search_bx = gr.Textbox(label="trial_searches", scale=0, min_width=110, visible=False)
                    count_bx = gr.Textbox(label="revision_number", scale=0, min_width=110, visible=False)
                
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
                        value="🔄 **Agent is processing...** Please wait while the evaluation is in progress.",
                        visible=True
                    )
                
                # Function to hide processing status
                def hide_processing():
                    return gr.update(visible=False)
                
                with gr.Accordion("Manage Agent", open=True):
                    checks = list(self.graph.nodes.keys())
                    checks.remove('__start__')
                    stop_after = gr.CheckboxGroup(checks,label="Interrupt After State", value=checks, scale=0, min_width=400)
                    with gr.Row():
                        thread_pd = gr.Dropdown(choices=self.threads,interactive=True, label="select thread", min_width=120, scale=0, visible=False)
                        step_pd = gr.Dropdown(choices=['N/A'],interactive=True, label="select step", min_width=160, scale=1, visible=False)
                
                # Add progress bar
                # progress_bar =  gr.Progress()
                
                with gr.Accordion("Live Agent Output", open=False):
                    live = gr.Textbox(label="", lines=10, max_lines=25)
        
                # actions
                sdisps =[prompt_bx,tab_notification,last_node,eligible_bx, nnode_bx,threadid_bx,count_bx,step_pd,thread_pd, search_bx]
                
                # Add debug mode toggle functionality
                debug_mode.change(
                    fn=toggle_debug_fields,
                    inputs=[debug_mode],
                    outputs=[threadid_bx, search_bx, count_bx, thread_pd, step_pd]
                )
                
                # sdisps =[prompt_bx,last_node,eligible_bx, nnode_bx,threadid_bx,revision_bx,count_bx,step_pd,thread_pd]
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                                fn=updt_disp, inputs=None, outputs=sdisps)
                step_pd.input(self.copy_state,[step_pd],None).then(
                              fn=updt_disp, inputs=None, outputs=sdisps)
                gen_btn.click(vary_btn,gr.Number("secondary", visible=False), gen_btn).then(
                              vary_btn,gr.Number("primary", visible=False), cont_btn).then(
                              fn=show_processing, inputs=None, outputs=processing_status).then(
                              fn=self.run_agent, inputs=[gr.Number(True, visible=False),prompt_bx,stop_after], outputs=[live],show_progress=True).then(
                              fn=hide_processing, inputs=None, outputs=processing_status).then(
                              fn=updt_disp, inputs=None, outputs=sdisps)
                cont_btn.click(fn=show_processing, inputs=None, outputs=processing_status).then(
                               fn=self.run_agent, inputs=[gr.Number(False, visible=False),prompt_bx,stop_after], 
                               outputs=[live]).then(
                               fn=hide_processing, inputs=None, outputs=processing_status).then(
                               fn=updt_disp, inputs=None, outputs=sdisps)            
        
            with gr.Tab("Profile"):
                # Add informative text at the top of the Profile tab
                profile_info = gr.Markdown(
                    value="""## 📋 Patient Profile Information

**🔄 Refresh the agent profile** - The profile is generated by a language model based on patient data.

**✏️ You can directly change the text and press the 'Modify' button**, especially in cases of:
- ⚠️ Conflicts with trial policies
- 🔍 When relevant trials cannot be found
- 📝 Need to adjust the profile for specific requirements

**💡 Tip:** The patient profile is crucial for matching appropriate clinical trials.""",
                    visible=True
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")

                profile = gr.Textbox(label="Patient's profile created from patient's data", lines=10, interactive=True)
                refresh_btn.click(fn=self.get_state, inputs=gr.Number("patient_profile", visible=False), outputs=profile)
                modify_btn.click(fn=self.modify_state, inputs=[gr.Number("patient_profile", visible=False),
                                                          gr.Number(self.SENTINEL_NONE, visible=False), profile],outputs=None).then(
                                 fn=updt_disp, inputs=None, outputs=sdisps)                                              

            with gr.Tab("Policies"):
                # Add informative text at the top of the Policies tab
                policies_info = gr.Markdown(
                    value="""## 📜 Trial Policies Review

**🔍 View retrieved policies** based on the patient's profile.

**When to use this tab:**
- **✅ Check policy requirements** that may affect trial eligibility
- **📖 Review detailed policy content** for better understanding
- **🔄 Refresh** to get the latest policy information

**Next steps:**
- If policies seem restrictive → Check **'Policy Issue'** tab
- If policies look compatible → Proceed to **'Matched Trials'** tab""",
                    visible=True
                )
                
                refresh_btn = gr.Button("Refresh")
                policies_bx = gr.Textbox(label="Retieved participation policies based on patient's profile", lines=40)
                refresh_btn.click(fn=self.get_content, inputs=gr.Number("policies", visible=False), outputs=policies_bx)

            with gr.Tab("Policy Issue"):
                # Add informative text at the top of the Policy Issue tab
                policy_issue_info = gr.Markdown(
                    value="""## ⚠️ Policy Conflict Resolution

**🚨 When you see this tab highlighted** - There's a policy conflict that needs attention.

**Your options:**
1. **🔄 Refresh** to review the specific policy issue
2. **⏭️ Skip the policy** if it's not applicable to this patient
3. **🔙 Go back to 'Profile' tab** to modify patient information

**When to skip:**
- Policy doesn't apply to patient's condition
- Policy is outdated or incorrectly matched
- Manual review determines patient should proceed

**💡 Alternative:** Modify the patient profile to better match policies.""",
                    visible=True
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    skip_btn = gr.Button("Skip the policy")
                policy_issue_bx = gr.Textbox(label="Policy Issue", lines=10, interactive=False)
                
                def skip_policy_and_notify():
                    """Skip the current policy and show confirmation message"""
                    # Skip the policy in the state
                    self.modify_state("policy_skip", self.SENTINEL_NONE, "")
                    # Return confirmation message
                    return gr.update(
                        label="Policy Skipped", 
                        value="The current policy is skipped for this patient. Please continue evaluation in the Agent tab."
                    )
                
                refresh_btn.click(fn=self.get_issue_policy, inputs=None, outputs=policy_issue_bx)
                skip_btn.click(fn=skip_policy_and_notify, inputs=None, outputs=policy_issue_bx).then(
                                fn=updt_disp, inputs=None, outputs=sdisps)
                

            with gr.Tab("Matched Trials"):
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
                trials_bx = gr.Dataframe(label="Retieved relevant trials based on patient's profile", wrap=True, interactive=True, max_height=1000)
                refresh_btn.click(fn=self.get_table, inputs=gr.Number("trials", visible=False), outputs=trials_bx)                
            
            with gr.Tab("Trials Scores"):
                # Add informative text at the top of the Trials Scores tab
                trials_scores_info = gr.Markdown(
                    value="""## 📊 Trial Eligibility Scores

**🏆 Final results** - Detailed scoring and ranking of matched trials.

**Understanding the scores:**
- **🥇 Higher scores** = Better match for the patient
- **📈 Multiple criteria** considered (eligibility, relevance, availability)
- **📋 Read-only view** - scoring is automatically calculated

**This is typically your final destination** in the evaluation process.

**If scores seem incorrect:**
- **🔙 Modify patient profile** in the 'Profile' tab
- **🔄 Re-run evaluation** from the 'Agent' tab""",
                    visible=True
                )
                
                # trials_scores_bx = gr.Textbox(label="Trials Scores based on patient's profile")
                trials_scores_bx = gr.Dataframe(label="Trials Scores based on patient's profile", wrap=True, interactive=False, max_height=1000)
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
