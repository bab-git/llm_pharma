import os

import gradio as gr
import pandas as pd


class trials_gui:
    SENTINEL_NONE = "__NONE__"

    # Data-driven mappings
    NODE_NOTIFICATIONS = {
        "patient_collector": "Check Patient Profile - Patient profile has been created",
        "policy_search": "Check Policy Conflict - Relevant Clinical Policies have been retrieved",
        "policy_evaluator": "Check Policy Conflict - In case of policy conflict, your action is needed",
        "trial_search": "Check Potential Trials - Potentially relevant clinical trials have been found",
        "grade_trials": "Check Trials Scores - Calculated Trial Relevance scores are ready",
        "profile_rewriter": "Go to Profile Tab - Patient profile has been updated",
    }

    APP_DESCRIPTION = """## 🏥 Clinical Trial Eligibility Assistant

**Purpose:** This AI-powered application helps healthcare professionals evaluate patient eligibility for clinical trials by analyzing patient data, reviewing trial policies, and matching patients with appropriate studies.

**How it works:** The system uses a multi-stage evaluation process to assess patient compatibility with available clinical trials, ensuring both patient safety and trial requirements are met."""

    def __init__(self, workflow_manager_or_graph, share=False):
        # Handle both WorkflowManager and compiled graph
        if hasattr(workflow_manager_or_graph, "app"):
            # It's a WorkflowManager
            self.workflow_manager = workflow_manager_or_graph
            self.graph = workflow_manager_or_graph.app
        else:
            # It's a compiled graph (demo mode)
            self.workflow_manager = None
            self.graph = workflow_manager_or_graph

        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 10
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.demo = self.create_interface()

    # === Helper Methods for State Updates ===

    def upd(self, key, *, label=None, lines=None, value=None):
        """Helper for common gr.update() patterns"""
        if value is not None:
            return gr.update(value=value, label=label, lines=lines)

        current_values = self.graph.get_state(self.thread)
        val = current_values.values.get(key, "") if current_values.values else ""

        if label is None and current_values.values:
            last_node, nnode, thread_id, rev, astep = self.get_disp_state()
            label = f"last_node: {last_node}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"

        return gr.update(value=val, label=label, lines=lines)

    def placeholder_for(self, key):
        """Get placeholder text for different components"""
        placeholders = {
            "patient_profile": "Patient profile will appear here after evaluation starts...",
            "policies": "Current policies will appear here after policy search...",
            "trials": "Potential trials will appear here after policy evaluation...",
            "policy_status": "Policy issues will appear here when conflicts are detected...",
            "stages_history": "Stages execution history will appear here as the agent runs...",
        }
        return placeholders.get(key, "No data available")

    def get_disp_state(self):
        """Get current display state information"""
        current_state = self.graph.get_state(self.thread)
        last_node = current_state.values.get("last_node", "")
        acount = current_state.values.get("revision_number", 0)
        rev = current_state.values.get("revision_number", 0)
        nnode = current_state.next
        return last_node, nnode, self.thread_id, rev, acount

    # === Notification Logic ===

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

        # policy_eligible = current_state.values.get("policy_eligible", None)
        trial_found = current_state.values.get("trial_found")
        trials = current_state.values.get("trials", [])

        # Special cases first
        if trial_found is True:
            return "🎉 Trial Scores - Perfectly Matched clinical trials have been found! 🎉"
        elif last_node == "grade_trials":
            return """⚠️ Trials Scores - No matched trials found. Please review the relevance scores for more details.
Your options:            
   A - Continue with auto-generated profile rewriter --> Continue Evaluation,
   B - Manually modify the patient profile to better match trials."""
        elif last_node == "trial_search" and trials == []:
            if nnode is None:
                return "Agent Tab - The pipeline couldn't find any potential/relevant trials. Try another patient."
        elif last_node == "profile_rewriter":
            return "Go to Profile Tab - The patient profile has been rewritten by the agent to increase the chances of finding relevant trials. You can also manually modify the patient profile."

        return self.NODE_NOTIFICATIONS.get(last_node, f"Agent is at: {last_node}")

    # === State Management ===

    def get_current_state_values(self):
        """Get current state values safely"""
        current_values = self.graph.get_state(self.thread)
        return current_values.values if current_values.values else {}

    def get_patient_profile_with_formatting(self):
        """Get patient profile with preserved textbox formatting (multi-line support)."""
        state_values = self.get_current_state_values()
        if "patient_profile" in state_values:
            return gr.update(value=state_values["patient_profile"], lines=5)
        else:
            return gr.update(value="", lines=5)

    def get_current_policies(self):
        """Get the current policies from the agent state."""
        state_values = self.get_current_state_values()

        if not state_values or "policies" not in state_values:
            return gr.update(
                label="📜 Policies Related to the Patient",
                value=self.placeholder_for("policies"),
            )

        if state_values["policies"]:
            policies = state_values["policies"]
            policies_text = "\n\n".join(
                f"**Policy {i+1}:**\n{doc.page_content}"
                for i, doc in enumerate(policies)
            )
            return gr.update(value=policies_text)
        else:
            return gr.update(
                label="📜 Policies Related to the Patient",
                value="No policies found in current state",
            )

    def get_last_policy_status(self):
        """Get the status of the last checked policy."""
        state_values = self.get_current_state_values()

        if not state_values or "last_node" not in state_values:
            return gr.update(
                label="Policy Status", value="No policy evaluation started yet"
            )

        policy_status = "No policy checked yet"

        if "checked_policy" in state_values and state_values["checked_policy"]:
            checked_policy = state_values["checked_policy"]
            policy_eligible = state_values.get("policy_eligible", None)
            rejection_reason = state_values.get("rejection_reason", "")

            # Get policy title/header
            policy_content = checked_policy.page_content
            policy_header = (
                policy_content.split("\n")[0] if policy_content else "Policy"
            )

            if policy_eligible is True:
                status_icon = "✅"
                status_text = "PASSED"
                policy_status = (
                    f"{status_icon} Last Policy: {policy_header}\nStatus: {status_text}"
                )
            elif policy_eligible is False:
                status_icon = "❌"
                status_text = "FAILED"
                policy_status = (
                    f"{status_icon} Last Policy: {policy_header}\nStatus: {status_text}"
                )

                if rejection_reason:
                    policy_status += f"\n\n🚨 **Rejection Reason:**\n{rejection_reason}"
            else:
                status_icon = "❓"
                status_text = "UNKNOWN"
                policy_status = (
                    f"{status_icon} Last Policy: {policy_header}\nStatus: {status_text}"
                )

        return gr.update(value=policy_status)

    def get_trials_summary_table(self):
        """Get a summary table of trials with only nctid, diseases, and relevance columns."""
        state_values = self.get_current_state_values()

        # Check if the state has been initialized
        if not state_values or "last_node" not in state_values:
            return pd.DataFrame(
                {
                    "nctid": ["No data available"],
                    "diseases": ["Start evaluation to see results"],
                    "relevance": ['Click "Start Evaluation" in the Agent tab'],
                }
            )

        # Check if we have relevant trials data (after grade_trials)
        if "relevant_trials" in state_values and state_values["relevant_trials"]:
            relevant_trials = state_values["relevant_trials"]
            trials_data = []

            for trial in relevant_trials:
                nctid = trial.get("nctid", "Unknown")
                # Get diseases from the original trials data if available
                diseases = "Unknown"
                if "trials" in state_values:
                    for orig_trial in state_values["trials"]:
                        if orig_trial.metadata.get("nctid") == nctid:
                            diseases = orig_trial.metadata.get("diseases", "Unknown")
                            break

                relevance = trial.get("relevance_score", "Unknown")
                trials_data.append(
                    {"nctid": nctid, "diseases": diseases, "relevance": relevance}
                )

            return pd.DataFrame(trials_data)

        # If no relevant trials yet, check if we have basic trials data
        elif "trials" in state_values and state_values["trials"]:
            trials = state_values["trials"]
            trials_data = []

            for trial in trials:
                nctid = trial.metadata.get("nctid", "Unknown")
                diseases = trial.metadata.get("diseases", "Unknown")
                trials_data.append(
                    {
                        "nctid": nctid,
                        "diseases": diseases,
                        "relevance": "Not graded yet",
                    }
                )

            return pd.DataFrame(trials_data)

        else:
            return pd.DataFrame(
                {
                    "nctid": ["No trials found yet"],
                    "diseases": ["Complete policy evaluation first"],
                    "relevance": ["Trials will be searched after policy check"],
                }
            )

    def get_stages_history(self):
        """Get the history of stages/nodes that have been executed."""
        stages_list = []

        try:
            # Get state history in reverse order (latest first)
            for state in self.graph.get_state_history(self.thread):
                if state.metadata.get("step", 0) < 1:  # Skip early states
                    continue

                last_node = state.values.get("last_node", "unknown")
                revision_number = state.values.get("revision_number", 0)
                thread_ts = state.config["configurable"].get("thread_ts", "unknown")

                # Create a stage entry
                stage_entry = (
                    f"Step {revision_number}: {last_node} (ts: {thread_ts[-8:]})"
                )
                stages_list.append(stage_entry)

            # Reverse to show chronological order (oldest first)
            stages_list.reverse()

            if not stages_list:
                return gr.update(
                    label="Stages History", value=self.placeholder_for("stages_history")
                )

            stages_text = "\n".join(stages_list)
            last_node, nnode, thread_id, rev, astep = self.get_disp_state()
            new_label = f"Stages History (Thread: {thread_id}, Current: {last_node})"

            return gr.update(label=new_label, value=stages_text)

        except Exception as e:
            return gr.update(
                label="Stages History (Error)",
                value=f"Error retrieving stages: {str(e)}",
            )

    def get_table(self, key=None):
        """Get table data for trials or scores"""
        state_values = self.get_current_state_values()

        # Check if the state has been initialized with required keys
        if not state_values or "last_node" not in state_values:
            # Return placeholder data when state is not initialized
            if key == "trials_scores":
                return pd.DataFrame(
                    {
                        "nctid": ["No data available"],
                        "relevance_score": ["Start evaluation to see results"],
                        "explanation": ["Please run the agent evaluation first"],
                        "further_information": [
                            'Click "Start Evaluation" in the Agent tab'
                        ],
                    }
                )
            elif key == "trials":
                return pd.DataFrame(
                    {
                        "index": ["No data available"],
                        "nctid": ["Start evaluation to see results"],
                        "diseases": ["Please run the agent evaluation first"],
                        "Criteria": ['Click "Start Evaluation" in the Agent tab'],
                    }
                )
            else:
                raise ValueError("key should be 'relevant_trials' or 'trials'")

        # last_node = state_values["last_node"]
        df = pd.DataFrame(columns=["no data available"])

        if key == "trials_scores":
            if "relevant_trials" in state_values:
                scores = state_values["relevant_trials"]
                df = pd.DataFrame(scores)
                df = df.reindex(
                    columns=[
                        "nctid",
                        "relevance_score",
                        "explanation",
                        "further_information",
                    ]
                )
            else:
                # Return placeholder when no trials have been scored yet
                df = pd.DataFrame(
                    {
                        "nctid": ["No trials scored yet"],
                        "relevance_score": ["Complete trial search first"],
                        "explanation": ["Trials need to be retrieved and graded"],
                        "further_information": ["Continue evaluation in the Agent tab"],
                    }
                )
        elif key == "trials":
            if "trials" in state_values:
                documents = state_values[key]
                data = []
                for idx, doc in enumerate(documents):
                    page_content = doc.page_content
                    nctid = doc.metadata["nctid"]
                    diseases = doc.metadata["diseases"]
                    data.append(
                        {
                            "index": idx,
                            "nctid": nctid,
                            "diseases": diseases,
                            "Criteria": page_content,
                        }
                    )
                df = pd.DataFrame(data)
            else:
                # Return placeholder when no trials have been retrieved yet
                df = pd.DataFrame(
                    {
                        "index": ["No trials retrieved yet"],
                        "nctid": ["Complete policy evaluation first"],
                        "diseases": ["Trials will be searched after policy check"],
                        "Criteria": ["Continue evaluation in the Agent tab"],
                    }
                )
        else:
            raise ValueError("key should be 'relevant_trials' or 'trials'")
        return df

    # === UI Component Builders ===

    def build_agent_tab(self):
        """Build the main agent control tab"""
        with gr.Tab("Agent"):
            # Concise agent tab explanation with two columns
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        value="""## 🤖 Agent Control Center

**Start Here:**
- ➡️ **Enter Patient Query**
- ➡️ **Click 'Start Evaluation'**
- ➡️ **Monitor Progress via Notifications**

**Tips:**
- 💡 **Follow Notifications:** They guide you through each step
- 💡 **Use 'Manage Agent'** for advanced options and configurations
""",
                        visible=True,
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        value="""## Process Overview:

- 🔍 **Profile Creation:** Generate a patient profile
- ⚠️ **Policy Check:** Identify and resolve conflicting clinical policies
- 🔄 **Trial Matching:** Find potential clinical trial matches
- 🎯 **Trial Relevance:** Determine relevant trials for the patient
""",
                        visible=True,
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
                        interactive=True,
                    )
                tab_notification = gr.Textbox(
                    value="Ready to start agent evaluation",
                    label="🔔 Your Notification Center 🔔",
                    lines=4,
                    interactive=False,
                    visible=True,
                    scale=3,
                    elem_id="notification_center",
                )

            # Add processing indicator
            processing_status = gr.Markdown(value="", visible=False)

            with gr.Row():
                gen_btn = gr.Button(
                    "Start Evaluation", scale=0, min_width=120, variant="primary"
                )
                cont_btn = gr.Button("Continue Evaluation", scale=0, min_width=120)
                debug_mode = gr.Checkbox(
                    label="🔧 Debug Mode", value=False, scale=0, min_width=120,
                )

            with gr.Row():
                last_node = gr.Textbox(label="Agent's last stop", min_width=150)
                eligible_bx = gr.Textbox(label="Is Patient Eligible?", min_width=50)
                nnode_bx = gr.Textbox(label="Agent's next step", min_width=150)
                threadid_bx = gr.Textbox(
                    label="Thread", scale=0, min_width=80, visible=False
                )
                search_bx = gr.Textbox(
                    label="trial_searches", scale=0, min_width=110, visible=False
                )
                count_bx = gr.Textbox(
                    label="revision_number", scale=0, min_width=110, visible=False
                )

            with gr.Accordion("Manage Agent", open=True):
                checks = list(self.graph.nodes.keys())
                checks.remove("__start__")
                checks_values = checks.copy()
                stop_after = gr.CheckboxGroup(
                    checks,
                    label="Interrupt After State",
                    value=checks_values,
                    scale=0,
                    min_width=400,
                )
                with gr.Row():
                    thread_pd = gr.Dropdown(
                        choices=self.threads,
                        interactive=True,
                        label="select thread",
                        min_width=120,
                        scale=0,
                        visible=False,
                    )
                    step_pd = gr.Dropdown(
                        choices=["N/A"],
                        interactive=True,
                        label="select step",
                        min_width=160,
                        scale=1,
                        visible=False,
                    )

            with gr.Accordion("Live Agent Output - for debugging", open=False):
                live = gr.Textbox(label="", lines=10, max_lines=25)

        return {
            "prompt_bx": prompt_bx,
            "patient_id_dropdown": patient_id_dropdown,
            "tab_notification": tab_notification,
            "processing_status": processing_status,
            "gen_btn": gen_btn,
            "cont_btn": cont_btn,
            "debug_mode": debug_mode,
            "last_node": last_node,
            "eligible_bx": eligible_bx,
            "nnode_bx": nnode_bx,
            "threadid_bx": threadid_bx,
            "search_bx": search_bx,
            "count_bx": count_bx,
            "stop_after": stop_after,
            "thread_pd": thread_pd,
            "step_pd": step_pd,
            "live": live,
        }

    def build_status_panels(self):
        """Build the four status panels under the agent tab"""
        # 1. Patient Profile Display
        with gr.Row(elem_id="profile-section"):
            with gr.Column(scale=3):
                profile_title = gr.Markdown(
                    value="## 📋 Current Patient Profile", visible=True
                )
                profile_help = gr.Markdown(
                    value="""**✏️ You can directly change the text and press the 'Modify' button**, especially in cases of:
- ⚠️ **Conflicts with trial policies**
- 🔍 **When relevant trials cannot be found**
- 📝 **Helping patient to match specific trial**""",
                    visible=False,
                )
            with gr.Column(scale=1):
                modify_profile_btn = gr.Button("✏️ Modify Profile", min_width=120, elem_classes=["my-special-btn"])
        with gr.Row(elem_id="profile-section"):
            current_profile = gr.Textbox(
                label="",
                lines=5,
                interactive=True,
                placeholder=self.placeholder_for("patient_profile"),
            )

        # 2. Policy Conflict Resolution (entire section in one container)
        with gr.Column(elem_id="policy-conflict-section"):
            with gr.Row():
                with gr.Column(scale=3):
                    policy_title = gr.Markdown(
                        value="## ⚠️ Policy Conflict Resolution", visible=True
                    )
                    policy_conflict_info = gr.Markdown(
                        value="""**🚨 Policy Conflict Detected**

- **⏭️ Skip if this policy is not relevant**
- **🔧 Modify patient profile if needed**
- **⏭️⏭️ Skip all policy checks for the patient**""",
                        visible=False,
                    )
                with gr.Column(scale=1):
                    policy_skip_btn = gr.Button("⏭️ Skip Policy", min_width=120, elem_classes=["my-special-btn"])
                    policy_big_skip_btn = gr.Button(
                        "⏭️ Skip All Policies", min_width=140, elem_classes=["my-special-btn"]
                    )
            with gr.Row():
                # Left column: Current Policies
                with gr.Column(scale=1):
                    current_policies = gr.Textbox(
                        label="📜 Policies Related to the Patient",
                        lines=15,
                        interactive=False,
                        placeholder=self.placeholder_for("policies"),
                    )
                # Right column: Policy Issues
                with gr.Column(scale=1):
                    policy_status = gr.Textbox(
                        label="⚠️ Policy Issues & Conflicts",
                        lines=4,
                        interactive=False,
                        placeholder=self.placeholder_for("policy_status"),
                    )

        # 3. Trials Summary Table
        with gr.Column(elem_id="trials-summary-section"):
            gr.Markdown(
                value="""## 🎯 Trials Summary (NCT ID | Diseases | Relevance)

You can obtain more information about each trial's details and possible relevance reasons in the **Potential Trials** and **Trials Scores** tabs."""
            )
            trials_summary = gr.Dataframe(
                label="",
                headers=["nctid", "diseases", "relevance"],
                interactive=False,
                wrap=True,
            )

        # 4. Stages History Box
        with gr.Row():
            stages_history = gr.Textbox(
                label="📈 Execution Stages History",
                lines=6,
                interactive=False,
                placeholder=self.placeholder_for("stages_history"),
            )

        return {
            "current_profile": current_profile,
            "modify_profile_btn": modify_profile_btn,
            "profile_help": profile_help,
            "profile_title": profile_title,
            "current_policies": current_policies,
            "policy_status": policy_status,
            "policy_conflict_info": policy_conflict_info,
            "policy_title": policy_title,
            "policy_skip_btn": policy_skip_btn,
            "policy_big_skip_btn": policy_big_skip_btn,
            "trials_summary": trials_summary,
            "stages_history": stages_history,
        }

    def build_other_tabs(self):
        """Build Potential Trials and Trials Scores tabs"""
        tabs_components = {}

        with gr.Tab("Potential Trials"):
            # Add informative text at the top of the Matched Trials tab
            gr.Markdown(
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
                visible=True,
            )

            with gr.Row():
                refresh_trials_btn = gr.Button("Refresh", elem_classes=["my-special-btn"])
            trials_bx = gr.Dataframe(
                label="Retrieved relevant trials based on patient's profile",
                wrap=True,
                interactive=True,
            )

            tabs_components["refresh_trials_btn"] = refresh_trials_btn
            tabs_components["trials_bx"] = trials_bx

        with gr.Tab("Trials Scores"):
            # Add informative text at the top of the Trials Scores tab
            gr.Markdown(
                value="""## 📊 Trial Eligibility Scores

**🏆 Final results** - Detailed scoring and ranking of matched trials.

**Understanding the scores:**
- **✅ Relevant (Yes):** The patient's profile meets the trial's inclusion criteria and does not meet any exclusion criteria.
- **❌ Not Relevant (No):** The patient's profile meets any exclusion criteria or does not meet the trial's inclusion diseases.
- **🔄 Refresh** to get updated trial matches

**This is typically your final destination** in the evaluation process.

**If scores seem incorrect:**
- **🔙 Modify patient profile** in the 'Profile' tab
- **🔄 Re-run evaluation** from the 'Agent' tab""",
                visible=True,
            )
            with gr.Row():
                refresh_scores_btn = gr.Button("Refresh", elem_classes=["my-special-btn"])
            trials_scores_bx = gr.Dataframe(
                label="Trials Scores based on patient's profile",
                wrap=True,
                interactive=False,
            )

            tabs_components["refresh_scores_btn"] = refresh_scores_btn
            tabs_components["trials_scores_bx"] = trials_scores_bx

        return tabs_components

    # === Callback Functions ===

    def update_prompt_from_patient_id(self, patient_selection):
        """Update the prompt text box when a patient ID is selected from dropdown"""
        if patient_selection and patient_selection.startswith("Patient "):
            patient_id = patient_selection.split(" ")[1]
            return f"Is patient_ID {patient_id} eligible for any medical trial?"
        return "Is patient_ID 56 eligible for any medical trial?"

    def toggle_debug_fields(self, debug_enabled):
        """Toggle visibility of debug fields"""
        return [
            gr.update(visible=debug_enabled),  # threadid_bx
            gr.update(visible=debug_enabled),  # search_bx
            gr.update(visible=debug_enabled),  # count_bx
            gr.update(visible=debug_enabled),  # thread_pd
            gr.update(visible=debug_enabled),  # step_pd
        ]

    def show_processing(self):
        """Show processing status"""
        return gr.update(
            value="🔄 **AGENT IS PROCESSING...** \n Please wait while the evaluation is in progress.",
            visible=True,
        )

    def hide_processing(self):
        """Hide processing status"""
        return gr.update(visible=False)

    def vary_btn(self, stat):
        """Change button variant"""
        return gr.update(variant=stat)

    def skip_policy_and_notify(self):
        """Skip the current policy and show confirmation message"""
        self.modify_state("policy_skip", self.SENTINEL_NONE, "")
        return gr.update(
            label="Policy Skipped",
            value="The current policy is skipped for this patient.\n\nPlease continue evaluation of remaining policies in the Agent tab.",
        )

    def big_skip_policy_and_notify(self):
        """Skip the whole policy check and show confirmation message"""
        self.modify_state("policy_big_skip", self.SENTINEL_NONE, "")
        return gr.update(
            label="Policy Skipped",
            value="✅ The 'policy check phase' is completely skipped for this patient.\n\nPlease continue the next phase, Trial searches, via the Agent tab.",
        )

    def refresh_all_status(self, skip_policy_status_update=False):
        """Refresh all the status components."""
        state_values = self.get_current_state_values()
        # last_node = state_values.get("last_node", "")
        profile_ready = (
            state_values
            and "patient_profile" in state_values
            and state_values["patient_profile"]
            and state_values["patient_profile"].strip()
        )
        policy_search_done = (
            state_values
            and "last_node" in state_values
            and state_values["last_node"]
            in [
                "policy_search",
                "policy_evaluator",
                "trial_search",
                "grade_trials",
                "profile_rewriter",
            ]
        )
        policy_conflict_buttons_visible = (
            state_values
            and "last_node" in state_values
            and state_values["last_node"] == "policy_evaluator"
            and (
                self.graph.get_state(self.thread).next is None
                or len(self.graph.get_state(self.thread).next) == 0
            )
        )
        # Show "Skip All Policies" button after policy search is done
        policy_big_skip_visible = policy_search_done

        return [
            self.get_patient_profile_with_formatting(),  # current_profile - preserve multi-line formatting
            self.get_current_policies(),  # current_policies - same as profile textbox behavior
            (
                self.get_last_policy_status()
                if not skip_policy_status_update
                else gr.update()
            ),  # policy_status
            self.get_trials_summary_table(),  # trials_summary
            self.get_stages_history(),  # stages_history
            gr.update(
                visible=profile_ready
            ),  # profile_help - show only when profile is ready
            gr.update(visible=True),  # profile_title - always visible
            gr.update(
                visible=policy_search_done
            ),  # policy_conflict_info - show only when policy search is done
            gr.update(visible=True),  # policy_title - always visible
            gr.update(
                visible=policy_conflict_buttons_visible
            ),  # policy_skip_btn - show only when policy conflict detected
            gr.update(
                visible=policy_big_skip_visible
            ),  # policy_big_skip_btn - show after policy search is done
        ]

    def updt_disp(self):
        """General update display on state change — now returns a 10-tuple in sdisps order."""
        current_state = self.graph.get_state(self.thread)
        hist = []
        for state in self.graph.get_state_history(self.thread):
            if state.metadata.get("step", 0) < 1:
                continue
            ts = state.config["configurable"].get("thread_ts", "unknown")
            tid = state.config["configurable"]["thread_id"]
            rev = state.values.get("revision_number", 0)
            ln = state.values.get("last_node", "")
            nn = state.next
            hist.append(f"{tid}:{rev}:{ln}:{nn}:{rev}:{ts}")

        # If no metadata yet, return defaults
        if not current_state.metadata:
            return (
                "",  # prompt_bx
                gr.update(
                    value="Ready to start agent evaluation", visible=True
                ),  # tab_notification
                "",  # last_node
                "",  # eligible_bx
                "",  # nnode_bx
                self.thread_id,  # threadid_bx
                "",  # count_bx
                gr.update(choices=["N/A"], value="N/A"),  # step_pd
                gr.update(choices=self.threads, value=self.thread_id),  # thread_pd
                "",  # search_bx
            )

        vals = current_state.values
        # 0) prompt
        prompt_val = vals.get("patient_prompt", "")
        # 1) notification
        notif = self.get_tab_notification(current_state)
        notif_upd = gr.update(value=notif, visible=True)
        # 2) last_node
        ln = vals.get("last_node", "").replace("_", " ").title()
        # 3) eligibility
        pe = vals.get("policy_eligible")
        if pe is True:
            elig = "✅ Yes"
        elif pe is False:
            elig = "❌ No"
        else:
            elig = "❓ Not determined"
        # 4) next_node
        nn = ""
        if current_state.next:
            nn = current_state.next[0].replace("_", " ").title()
        elif current_state.next is None:
            nn = "END"
        # 5) thread id
        tid = self.thread_id
        # 6) revision count
        cnt = vals.get("revision_number", "")
        # 7) step_pd update
        step_upd = gr.update(choices=hist, value=hist[0] if hist else None)
        # 8) thread_pd update
        thread_upd = gr.update(choices=self.threads, value=self.thread_id)
        # 9) search_bx
        search_cnt = vals.get("trial_searches", "")

        return (
            prompt_val,
            notif_upd,
            ln,
            elig,
            nn,
            tid,
            cnt,
            step_upd,
            thread_upd,
            search_cnt,
        )

    # === Agent Execution ===

    def run_agent(self, start, patient_prompt, stop_after):
        """Main agent execution function"""
        if start:
            self.iterations.append(0)
            # Get the current selected model from the state
            current_values = self.graph.get_state(self.thread)
            selected_model = (
                current_values.values.get("selected_model", "llama-3.3-70b-versatile")
                if current_values.values
                else "llama-3.3-70b-versatile"
            )

            config = {
                "patient_prompt": patient_prompt,
                "max_revisions": 3,
                "revision_number": 0,
                "trial_searches": 0,
                "max_trial_searches": 2,
                "last_node": "",
                "selected_model": selected_model,
            }
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
        else:
            config = None
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            # Use graph for execution (works for both WorkflowManager and demo)
            if config:
                # Initial run with config
                self.response = self.graph.invoke(config, self.thread)
            else:
                # Continue execution
                self.response = self.graph.invoke(None, self.thread)

            self.iterations[self.thread_id] += 1
            self.partial_message += str(self.response)
            self.partial_message += f"\n {'='*40}\n\n"
            last_node, nnode, _, rev, acount = self.get_disp_state()

            # Default: don't update
            # policies_update = gr.update()
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
                trials_update,  # trials_summary (must be a DataFrame or gr.update())
            )
            config = None
            if not nnode:
                return
            if last_node in stop_after:
                return
        return

    # === State Modification ===

    def modify_state(self, key, asnode, new_value):
        """Modify agent state"""
        if asnode == self.SENTINEL_NONE:
            asnode = None
        change_list = None
        current_states = list(self.graph.get_state_history(self.thread))
        last_node = current_states[0].values["last_node"]

        if key == "patient_profile":
            if last_node == "patient_collector":
                print("patient_collector node")
                asnode = None
                i_state = 0
            elif last_node == "policy_search":
                print("policy_search node")
                asnode = "policy_search"
                i_state = 0
            elif last_node == "policy_evaluator":
                asnode = "policy_evaluator"
                i_state = 1
                change_list = [("policy_eligible", "N/A")]
            elif last_node == "grade_trials":
                asnode = "trial_search"
                i_state = 1
            elif last_node == "profile_rewriter":
                asnode = "profile_rewriter"
                i_state = 0
            else:
                raise ValueError(f"unexpected last node {last_node}")
            current_values = current_states[i_state]
            current_values.values[key] = new_value
        elif key == "policy_skip":
            current_values = current_states[1]
            change_list = [("policy_eligible", True), ("rejection_reason", "N/A")]
            current_values.values["unchecked_policies"].pop(0)
            asnode = "policy_evaluator"
        elif key == "policy_big_skip":
            current_values = current_states[1]
            change_list = [("policy_eligible", True), ("rejection_reason", "N/A")]
            current_values.values["unchecked_policies"] = []
            asnode = "policy_evaluator"
        else:
            raise ValueError(f"unexpected key {key}")

        if change_list is not None:
            for key_ext, val_ext in change_list:
                current_values.values[key_ext] = val_ext

        self.graph.update_state(self.thread, current_values.values, as_node=asnode)
        print("State updated")
        current_state = list(self.graph.get_state_history(self.thread))[0]
        print(current_state.next)
        return

    def switch_thread(self, new_thread_id):
        """Switch to a different thread"""
        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id
        return

    def find_config(self, thread_ts):
        """Find config by thread timestamp"""
        for state in self.graph.get_state_history(self.thread):
            config = state.config
            if config["configurable"]["thread_ts"] == thread_ts:
                return config
        return None

    def copy_state(self, hist_str):
        """Copy an old state to current state"""
        thread_ts = hist_str.split(":")[-1]
        config = self.find_config(thread_ts)
        state = self.graph.get_state(config)
        self.graph.update_state(
            self.thread, state.values, as_node=state.values["last_node"]
        )
        new_state = self.graph.get_state(self.thread)  # should now match
        new_thread_ts = new_state.config["configurable"]["thread_ts"]
        # tid = new_state.config["configurable"]["thread_id"]
        revision_number = new_state.values["revision_number"]
        last_node = new_state.values["last_node"]
        rev = new_state.values["revision_number"]
        nnode = new_state.next
        return last_node, nnode, new_thread_ts, rev, revision_number

    # === Main Interface Creation ===

    def create_interface(self):
        """Create the main Gradio interface"""
        with gr.Blocks(
            theme=gr.themes.Default(spacing_size="sm", text_size="lg"),
            css="""
            /* Darker background for the entire profile section */
            #profile-section {
                background-color: #2a3f54;  /* a deep slate blue */
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }

            /* target by ID */
            #notification_center textarea {
                background-color: #b3d1ea;   /* less light, medium blue */
                border: 2px solid #4a90e2;
                color: #333;
                font-size: 22px !important;
            }
            .main-container, .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
                padding: 20px !important;
            }
            .gradio-container, .main-container, body, html {
                font-size: 22px !important;
            }
            .gr-textbox label, .gr-textbox textarea {
                font-size: 22px !important;
            }
            /* Make the font of the profile textbox bigger */
            #profile-section textarea {
               font-size: 20px !important;
            }
            /* Add a background for the Policy Conflict Resolution section */
            #policy-conflict-section {
                background-color: #4B0000; /* very dark red */
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            /* Add a background for the Trials Summary section */
            #trials-summary-section {
                background-color: #184d27; /* dark green */
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            /* style all buttons with the class 'my-special-btn' */
            .my-special-btn {
            background-color: #FFD700 !important;
            color: #000 !important;
            }
            """,
        ) as demo:

            # Add app description at the very top
            # app_description = gr.Markdown(value=self.APP_DESCRIPTION, visible=True)

            # Build all UI components
            agent_components = self.build_agent_tab()
            status_components = self.build_status_panels()
            tabs_components = self.build_other_tabs()

            # Centralize all component references
            all_components = {
                **agent_components,
                **status_components,
                **tabs_components,
            }

            # Wire up all interactions
            self.bind_callbacks(all_components)

        return demo

    def bind_callbacks(self, components):
        """Centrally bind all callbacks to components"""
        # Extract specific components for clarity
        c = components  # shorthand

        # Status display components list
        sdisps = [
            c["prompt_bx"],
            c["tab_notification"],
            c["last_node"],
            c["eligible_bx"],
            c["nnode_bx"],
            c["threadid_bx"],
            c["count_bx"],
            c["step_pd"],
            c["thread_pd"],
            c["search_bx"],
        ]

        # Status panel components list
        status_components = [
            c["current_profile"],
            c["current_policies"],
            c["policy_status"],
            c["trials_summary"],
            c["stages_history"],
            c["profile_help"],
            c["profile_title"],
            c["policy_conflict_info"],
            c["policy_title"],
            c["policy_skip_btn"],
            c["policy_big_skip_btn"],
        ]

        # Debug mode toggle
        c["debug_mode"].change(
            fn=self.toggle_debug_fields,
            inputs=[c["debug_mode"]],
            outputs=[
                c["threadid_bx"],
                c["search_bx"],
                c["count_bx"],
                c["thread_pd"],
                c["step_pd"],
            ],
        )

        # Patient ID dropdown
        c["patient_id_dropdown"].change(
            fn=self.update_prompt_from_patient_id,
            inputs=[c["patient_id_dropdown"]],
            outputs=[c["prompt_bx"]],
        )

        # Profile modification
        c["modify_profile_btn"].click(
            fn=self.modify_state,
            inputs=[
                gr.Number("patient_profile", visible=False),
                gr.Number(self.SENTINEL_NONE, visible=False),
                c["current_profile"],
            ],
            outputs=None,
        ).then(fn=self.updt_disp, inputs=None, outputs=sdisps).then(
            fn=self.refresh_all_status, inputs=None, outputs=status_components
        )

        # Policy skip buttons
        c["policy_skip_btn"].click(
            fn=self.skip_policy_and_notify, inputs=None, outputs=c["policy_status"]
        ).then(fn=self.updt_disp, inputs=None, outputs=sdisps).then(
            fn=self.refresh_all_status, inputs=None, outputs=status_components
        )

        c["policy_big_skip_btn"].click(
            fn=self.big_skip_policy_and_notify, inputs=None, outputs=c["policy_status"]
        ).then(fn=self.updt_disp, inputs=None, outputs=sdisps).then(
            fn=lambda: self.refresh_all_status(skip_policy_status_update=True),
            inputs=None,
            outputs=status_components,
        )

        # Thread and step controls
        c["thread_pd"].input(self.switch_thread, [c["thread_pd"]], None).then(
            fn=self.updt_disp, inputs=None, outputs=sdisps
        ).then(fn=self.refresh_all_status, inputs=None, outputs=status_components)

        c["step_pd"].input(self.copy_state, [c["step_pd"]], None).then(
            fn=self.updt_disp, inputs=None, outputs=sdisps
        ).then(fn=self.refresh_all_status, inputs=None, outputs=status_components)

        # Main agent buttons
        c["gen_btn"].click(
            self.vary_btn, gr.Number("secondary", visible=False), c["gen_btn"]
        ).then(self.vary_btn, gr.Number("primary", visible=False), c["cont_btn"]).then(
            fn=self.show_processing, inputs=None, outputs=c["processing_status"]
        ).then(
            fn=self.run_agent,
            inputs=[gr.Number(True, visible=False), c["prompt_bx"], c["stop_after"]],
            outputs=[c["live"], c["policy_status"], c["trials_summary"]],
            show_progress=True,
        ).then(
            fn=self.hide_processing, inputs=None, outputs=c["processing_status"]
        ).then(
            fn=self.updt_disp, inputs=None, outputs=sdisps
        ).then(
            fn=self.refresh_all_status, inputs=None, outputs=status_components
        )

        c["cont_btn"].click(
            fn=self.show_processing, inputs=None, outputs=c["processing_status"]
        ).then(
            fn=self.run_agent,
            inputs=[gr.Number(False, visible=False), c["prompt_bx"], c["stop_after"]],
            outputs=[c["live"], c["policy_status"], c["trials_summary"]],
        ).then(
            fn=self.hide_processing, inputs=None, outputs=c["processing_status"]
        ).then(
            fn=self.updt_disp, inputs=None, outputs=sdisps
        ).then(
            fn=self.refresh_all_status, inputs=None, outputs=status_components
        )

        # Refresh buttons in other tabs
        c["refresh_trials_btn"].click(
            fn=self.get_table,
            inputs=gr.Number("trials", visible=False),
            outputs=c["trials_bx"],
        )

        c["refresh_scores_btn"].click(
            fn=self.get_table,
            inputs=gr.Number("trials_scores", visible=False),
            outputs=c["trials_scores_bx"],
        )

    def launch(self, share=None):
        """Launch the Gradio interface"""
        if port := os.getenv("PORT1"):
            self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
        else:
            self.demo.launch(share=self.share)
