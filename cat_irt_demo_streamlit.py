"""
Demo Streamlit app for Computerized Adaptive Testing (CAT) using Item Response Theory (IRT)
to measure the personality trait of learning agility. This application allows users to
select a prior distribution, respond to a series of adaptive items, and visualize both
item characteristic/information curves and the evolving posterior distribution of
the latent trait (θ).

Author: Ludek Stehlik
Email: ludek.stehlik@gmail.com
GitHub: https://github.com/lstehlik2809
LinkedIn: https://www.linkedin.com/in/ludekstehlik/
"""


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CAT & IRT Demo", layout="wide")

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
# set of illustrative items for measuring learning agility
def load_item_bank() -> pd.DataFrame:
    """Returns the demo item bank with learning agility items."""
    return pd.DataFrame(
        {
            "item_id": [f"Q{i}" for i in range(1, 25)],
            "a": [  # Discrimination parameters 
                0.42, 0.43, 0.80, 0.70, 0.43, 0.72, 0.63, 0.71, 0.59, 0.80, 0.79, 0.27,
                0.55, 0.86, 0.92, 0.52, 0.61, 0.40, 0.48, 0.62, 0.93, 0.63, 0.50, 0.65,
            ],
            "b": [  # Difficulty parameters 
                0.73, -1.65, -1.08, -0.49, -1.74, 1.20, 1.06, -1.29, -0.59, -0.01, 0.27,
                0.16, -1.76, 0.20, 0.74, -1.10, 2.08, 0.39, 0.10, 1.60, 0.40, 2.01, -0.81, -0.99,
            ],
            "c": [0.0] * 24,  # Guessing parameters (set to 0 for this demo)
            "text": [ 
                "Do you actively seek out tasks or projects that require you to learn completely new skills?",  # b=0.73
                "Do you believe that learning new things is generally a positive experience?",  # b=-1.65
                "Are you open to trying new methods even if the old ones work reasonably well?",  # b=-1.08
                "Do you generally enjoy the process of figuring out something new?",  # b=-0.49
                "Do you agree that it's important to keep up with new developments in your field of interest?",  # b=-1.74
                "Do you thrive in situations where you have to quickly master unfamiliar concepts with little guidance?",  # b=1.20
                "Can you quickly adapt and apply your knowledge effectively when faced with unexpected challenges or novel problems?",  # b=1.06
                "Do you often find yourself curious about how things work?",  # b=-1.29
                "When you learn something new, do you usually try to apply it soon after?",  # b=-0.59
                "Do you consciously reflect on your experiences to identify lessons learned?",  # b=-0.01
                "Are you comfortable experimenting with new approaches, even if there's a risk of initial failure?",  # b=0.27
                "Do you actively look for opportunities to expand your knowledge and skills?",  # b=0.16
                "Are you generally open to hearing new ideas or perspectives?",  # b=-1.76
                "When you make a mistake, do you focus on what you can learn from it?",  # b=0.20
                "Are you confident in your ability to learn and master complex new subjects quickly?", # b=0.74
                "Do you pay attention when someone is explaining something new to you?",  # b=-1.10
                "Do you deliberately put yourself in highly complex and unfamiliar situations to accelerate your learning, even if it's very demanding?",  # b=2.08
                "Do you actively seek out feedback on your performance to identify areas for development?",  # b=0.39
                "When faced with a problem you don't know how to solve, are you resourceful in finding the necessary information or skills?",  # b=0.10
                "Do you often challenge conventional wisdom or established methods based on new insights you've gained?",  # b=1.60
                "Do you enjoy tackling problems that require you to acquire new knowledge or skills?",  # b=0.40
                "Are you driven to explore and master entirely new domains of knowledge, even if they are completely outside your current expertise and comfort zone?",  # b=2.01
                "After learning something new, do you often think about how it connects to what you already know?",  # b=-0.81
                "Are you willing to ask questions when you don't understand something new?",  # b=-0.99
            ],
        }
    )

# alternative set of illustrative items for measuring learning agility
# def load_item_bank() -> pd.DataFrame:
#     """Return the Learning-Agility demo item bank."""
#     return pd.DataFrame(
#         {
#             "item_id": [f"Q{i}" for i in range(1, 25)],
#             "a": [ # Discrimination parameters 
#                 0.42, 0.43, 0.80, 0.70, 0.43, 0.72, 0.63, 0.71, 0.59, 0.80, 0.79, 0.27,
#                 0.55, 0.86, 0.92, 0.52, 0.61, 0.40, 0.48, 0.62, 0.93, 0.63, 0.50, 0.65,
#             ],
#             "b": [ # Difficulty parameters 
#                 0.73, -1.65, -1.08, -0.49, -1.74, 1.20, 1.06, -1.29, -0.59, -0.01, 0.27,
#                 0.16, -1.76, 0.20, 0.74, -1.10, 2.08, 0.39, 0.10, 1.60, 0.40, 2.01, -0.81, -0.99,
#             ],
#             "c": [0.0] * 24,
#             "text": [ # Guessing parameters (set to 0 for this demo)
#                 "Do you quickly pick up a completely new software tool with minimal guidance?",
#                 "Do you enjoy striking up conversations on unfamiliar topics purely to learn something new?",
#                 "Do you often look up an explanation the moment you stumble across a concept you don’t know?",
#                 "Do you regularly experiment with small tweaks to streamline how you work?",
#                 "After finishing a task, do you automatically reflect on what you could learn from the experience?",
#                 "Have you ever taught yourself a complex skill (e.g., a programming language) in under a week to meet an urgent need?",
#                 "Can you perform effectively in a role where you had no prior experience after only a brief orientation?",
#                 "Do you sometimes read articles far outside your field simply out of curiosity?",
#                 "Do you frequently switch between very different tasks without losing momentum or accuracy?",
#                 "Do you anticipate future changes in your industry and start learning the required skills early?",
#                 "Would you say you actually enjoy steep learning curves?",
#                 "Do you proactively seek candid feedback to speed up your learning?",
#                 "Do you find yourself asking “why?” or “how?” several times a day?",
#                 "Are you often driven by the excitement of filling gaps in your knowledge?",
#                 "Do you deliberately choose projects that force you into unfamiliar territory?",
#                 "Do you feel a surge of energy when confronted with a brand-new challenge?",
#                 "Have you successfully applied knowledge from one discipline to solve a critical problem in a completely different field?",
#                 "Do you readily change your learning strategy when your first approach proves ineffective?",
#                 "Do you find it hard to ignore opportunities to learn something new, even when busy?",
#                 "Can you quickly synthesise information from several complex sources under intense time pressure?",
#                 "Would you describe yourself as thriving in ambiguous situations that require rapid learning?",
#                 "Have you ever been commended for mastering a difficult subject faster than expected?",
#                 "Do new ideas sometimes keep you awake at night because you’re eager to explore them?",
#                 "Do you often seek out knowledgeable peers for tips that will accelerate your learning curve?",
#             ],
#         }
#     )

item_bank = load_item_bank()

# --------------------------------------------------
# GLOBAL CSS – tighter column padding and chart legends
# --------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="column"] > div:first-child {
        padding-left: 0.15rem !important;
        padding-right: 0.15rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# IRT UTILITIES
# --------------------------------------------------
THETA_GRID = np.linspace(-4, 4, 801)

def icc(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

def iic(theta, a, b, c):
    p = icc(theta, a, b, c)
    dp = (1 - c) * a * np.exp(-a * (theta - b)) / (1 + np.exp(-a * (theta - b))) ** 2
    return dp**2 / (p * (1 - p))

def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def update_posterior(prior, a, b, c, resp):
    like = icc(THETA_GRID, a, b, c) if resp == 1 else 1 - icc(THETA_GRID, a, b, c)
    post = prior * like
    return post / post.sum()

def choose_next_item(bank, asked, theta_hat):
    remaining = bank[~bank.item_id.isin(asked)]
    infos = iic(theta_hat, remaining["a"].values, remaining["b"].values, remaining["c"].values)
    return remaining.iloc[int(np.argmax(infos))]

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.prior_choice = None
    st.session_state.posterior = None
    st.session_state.asked_items = []
    st.session_state.responses = []
    st.session_state.theta_history = []
    st.session_state.current_item = None
    st.session_state.test_finished = False

def reset_test():
    for k in [
        "prior_choice", "posterior", "asked_items", "responses", "theta_history", "current_item",
    ]:
        st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else None
    st.session_state.test_finished = False

# --------------------------------------------------
# CAT PAGE
# --------------------------------------------------

def render_cat_page():
    st.header("Computerized Adaptive Testing Demo")

    # ----- Prior selection / Question -----
    if st.session_state.prior_choice is None:
        st.subheader("First, choose prior distribution for ability/trait (*θ*)")
        pc = st.radio("Prior for *θ*", ("Flat (Uniform)", "Normal (*μ*=0, *σ*=1)"))
        if st.button("Start Test"):
            st.session_state.prior_choice = pc
            prior = np.ones_like(THETA_GRID) if pc.startswith("Flat") else normal_pdf(THETA_GRID)
            prior /= prior.sum()
            st.session_state.posterior = prior
            theta0 = float(np.average(THETA_GRID, weights=prior))
            st.session_state.theta_history.append(theta0)
            st.session_state.current_item = choose_next_item(item_bank, [], theta0)
            st.rerun()
        return

    if not st.session_state.test_finished:
        itm = st.session_state.current_item
        st.subheader(f"Respond to item {itm.item_id}")
        st.markdown(f"##### {itm.text}")
        yes_col, no_col = st.columns([1, 1], gap="small")
        resp = None
        if yes_col.button("Agree 👍", key=f"agree_{len(st.session_state.asked_items)}"):
            resp = 1
        if no_col.button("Disagree 👎", key=f"disagree_{len(st.session_state.asked_items)}"):
            resp = 0
        if resp is not None:
            st.session_state.responses.append(resp)
            st.session_state.asked_items.append(itm.item_id)
            st.session_state.posterior = update_posterior(st.session_state.posterior, itm.a, itm.b, itm.c, resp)
            theta_hat = float(np.average(THETA_GRID, weights=st.session_state.posterior))
            st.session_state.theta_history.append(theta_hat)
            st.session_state.test_finished = len(st.session_state.asked_items) == len(item_bank)
            if not st.session_state.test_finished:
                st.session_state.current_item = choose_next_item(item_bank, st.session_state.asked_items, theta_hat)
            st.rerun()
        st.button("Reset Test", on_click=reset_test, type="secondary")
    else:
        st.success("Test finished – all items administered.")
        st.button("Restart", on_click=reset_test)

    # ----- Charts side-by-side -----
    if st.session_state.posterior is not None or st.session_state.current_item is not None:
        col_item, col_post = st.columns(2, gap="medium")
        if st.session_state.current_item is not None:
            with col_item:
                itm = st.session_state.current_item
                th = np.linspace(-4, 4, 400)
                fig_i, ax1 = plt.subplots(figsize=(4, 3))
                l_icc, = ax1.plot(th, icc(th, itm.a, itm.b, itm.c), label="ICC")
                ax2 = ax1.twinx()
                l_iic, = ax2.plot(th, iic(th, itm.a, itm.b, itm.c), "--", label="IIC")
                ax1.set_xlabel("Ability/Trait Level (θ)"); ax1.set_ylabel("P(agree)"); ax2.set_ylabel("Information")
                ax1.set_title(f"ICC & IIC – {itm.item_id}")
                fig_i.legend(
                    [l_icc, l_iic],
                    ["ICC", "IIC"],
                    loc="lower center",
                    ncol=2,
                    bbox_to_anchor=(0.5, -0.265),
                    frameon=False
                )
                fig_i.tight_layout()
                fig_i.subplots_adjust(bottom=0.025)
                st.pyplot(fig_i)
        if st.session_state.posterior is not None:
            with col_post:
                posterior = st.session_state.posterior          # Already normalised
                cdf = posterior.cumsum()                 # Empirical CDF on the same grid

                # ---------- 95 % equal-tail credible interval ----------
                lo_idx = np.searchsorted(cdf, 0.025)
                hi_idx = np.searchsorted(cdf, 0.975)
                lo, hi = THETA_GRID[lo_idx], THETA_GRID[hi_idx]

                theta_hat = float(np.average(THETA_GRID, weights=posterior))  # mean (you could also use MAP)

                fig_p, ax_p = plt.subplots(figsize=(4, 3))

                # Posterior density
                ax_p.plot(THETA_GRID, posterior, zorder=2)

                # Shade only the 95 % CI
                mask = (THETA_GRID >= lo) & (THETA_GRID <= hi)
                ax_p.fill_between(
                    THETA_GRID[mask],
                    posterior[mask],
                    alpha=0.25,
                    zorder=1,
                    label="95% CrI",
                )

                # Vertical guide at θ̂
                ax_p.axvline(
                    theta_hat,
                    linestyle="--",
                    linewidth=1.4,
                    label=fr"$\hat{{\theta}} = {theta_hat:.2f}$",
                )

                # Axis labels & title
                ax_p.set_xlabel("Ability/Trait Level (θ)")
                ax_p.set_ylabel("Density")
                ax_p.set_title("Current Posterior Distribution of θ") 

                # Place the legend in that new space
                ax_p.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.3),
                    ncol=2,
                    frameon=False,
                )

                # Add extra canvas below the axes
                fig_p.tight_layout()
                fig_p.subplots_adjust(bottom=0) 
                st.pyplot(fig_p)

    st.markdown(
        """
        The chart on the left displays the psychometric properties for the selected item above, as modeled by the **Three-Parameter Logistic (3PL) model**. The **Item Characteristic Curve** (ICC) shows the probability of endorsing the item at different trait levels. The **Item Information Curve** (IIC) indicates the item's precision for measuring those trait levels (for more details, see the *Item Bank* page). The chart on the right shows the current **posterior distribution** of an individual's ability/trait level (*θ*), estimated from their responses. As responses are gathered, this distribution refines, with the mean of this posterior representing the expected trait level, and the shaded area indicating the 95% Credible Interval for this estimate (for more details, see the *How CAT Works* page).
        """
    )

# --------------------------------------------------
# ITEM BANK PAGE
# --------------------------------------------------

def render_bank_page():
    st.header("Item Bank")

    st.markdown(
        """
        Twenty-four items designed to measure the personality trait of [learning agility](https://psycnet.apa.org/buy/2022-19273-004) (the ability and willingness to rapidly learn from experiences and adapt that knowledge to navigate new or challenging situations effectively). <span style="color:red"><strong>However, it's important to note that these items are illustrative and have not been validated as reliable measures of learning agility.</strong></span>
        """,
        unsafe_allow_html=True
    )

    # ---------- TABLE ----------
    st.dataframe(
        item_bank.style.format(precision=2),
        use_container_width=True, 
        hide_index=True
    )

    st.divider()

    # ---------- TEXT + DROP‑DOWN + CHART ----------
    text_col, sel_plot_col = st.columns([2, 2], gap="medium")  

    with text_col:
        st.markdown(
        """
        **Item Characteristic Curve (ICC) and Item Information Curve (IIC)**
        
        The plots illustrate how individual items behave under the assumptions of [Item Response Theory (IRT)](https://en.wikipedia.org/wiki/Item_response_theory), specifically using the 3-Parameter Logistic (3PL) model with the **guessing parameter (*c*)** set to 0, effectively reducing it to a 2-Parameter Logistic (2PL) model.        
        
        The **ICC** shows the probability of a correct response (or response in the diagnostically expected direction) as a function of the examinee’s latent ability/trait (*θ*). The curve’s steepness is determined by the **discrimination parameter (*a*)**, while its central point reflects the **difficulty parameter (*b*)**—the ability level where the probability of success is 0.5.
        
        The **IIC** reflects how much **information** the item provides at each ability/trait level. The peak of the IIC occurs near the item’s difficulty (*b*) and is influenced by its discrimination (*a*). A more discriminating item provides more information and results in a narrower, taller peak.

        In adaptive testing, IICs are crucial: **items are selected not for overall difficulty, but for how informative they are at the examinee's current estimated ability/trait**.
        """
        )


    with sel_plot_col:
        sel_id = st.selectbox("Select item", item_bank.item_id.tolist(), key="sel_item_bank")
        sel = item_bank[item_bank.item_id == sel_id].iloc[0]
        st.write(f"**Item text:** *{sel.text}*")
        th = np.linspace(-4, 4, 400)
        fig_s, ax1 = plt.subplots(figsize=(6, 3))
        l_icc, = ax1.plot(th, icc(th, sel.a, sel.b, sel.c), label="ICC")
        ax1.set_xlabel("Ability/Trait Level (θ)"); ax1.set_ylabel("P(agree)")
        ax2 = ax1.twinx()
        l_iic, = ax2.plot(th, iic(th, sel.a, sel.b, sel.c), "--", label="IIC")
        ax2.set_ylabel("Information")
        ax1.set_title(f"ICC & IIC – {sel.item_id}")
        fig_s.legend(
            [l_icc, l_iic],
            ["ICC", "IIC"],
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.25),
            frameon=False
        )
        fig_s.tight_layout(); fig_s.subplots_adjust(bottom=0.05)
        st.pyplot(fig_s)

# --------------------------------------------------
# PAGE 3 – EXPLANATION (“How CAT Works”)
# --------------------------------------------------

def render_explain_page() -> None:
    st.header("How Computerized Adaptive Testing Works")

    st.markdown(
    """
    ### Key Ideas

    * **[Item Response Theory (IRT)](https://en.wikipedia.org/wiki/Item_response_theory)** models how individuals respond to questions based on both item characteristics and their own latent ability/trait (*θ*). In the **3-parameter logistic (3PL)** model, each dichotomic (True/False, Agree/Disagree) item is defined by three following parameters:
        - **Discrimination (*a*)** – how well the item distinguishes between low- and high-ability/trait individuals.  
        - **Difficulty (*b*)** – the ability/trait level where the item has a 50% chance of being answered correctly or in the diagnostically expected direction.  
        - **Guessing (*c*)** – the lower bound on the probability of a correct response or a response in the diagnostically expected direction (e.g. due to guessing).
    * **Information**: Each item provides the most information around its own difficulty. Items far from a person's θ contribute little—essentially, they’re inefficient.
    * **Adaptivity**: After each response, we update our estimate of *θ* and select the next most informative item given that current estimate. 
    * **[Computerized Adaptive Testing (CAT)](https://en.wikipedia.org/wiki/Computerized_adaptive_testing)**: A testing method where the selection of items is dynamically tailored to the individual's estimated ability/trait level. The goal is to efficiently and precisely measure the ability/trait.

    ---

    ### The Test Loop

    | Step | What Happens | In The App |
    |------|--------------|-------------|
    | 0 | Choose a prior distribution for ability/trait *θ* | Flat prior or normal distribution *N*(0, 1) |
    | 1 | Select item with max information at current *θ*| `choose_next_item()` |
    | 2 | Present item and record binary response | Buttons: *Agree* / *Disagree* |
    | 3 | Update the posterior of *θ* using [Bayes' Rule](https://en.wikipedia.org/wiki/Bayes%27_theorem) | `update_posterior()` |
    | 4 | Estimate *θ* from posterior (mean or [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)) | `np.average(THETA_GRID, weights=posterior)` |
    | 5 | Repeat steps 1–4 until a stopping rule is met | See below |

    ---

    ### Common Stopping Criteria

    * **Fixed-length** – Stop after a set number of items.  
      *Example:* Stop after 20 items.
    * **Measurement precision** – Stop when the posterior standard error (SEM) or credible interval width is below a threshold.  
      *Example:* Stop when SEM < 0.30.
    * **Information gain** – Stop if the expected gain from another item is negligible.  
      *Example:* Stop if adding any remaining item increases Fisher information by < 0.05.
    * **Time limit** – Optional cap on total administration time.  
      *Example:* Stop after 15 minutes have elapsed.
    * **Item pool exhausted** – Unlikely in practice, but serves as a safeguard (used here for demo).  
      *Example:* Stop if all 24 items have been presented.
    * **Combination of criteria** – In practice, it’s common to use two or more rules simultaneously.  
      *Example:* Stop when **either** 20 items have been administered **or** SEM < 0.30, whichever comes first.

    """
)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.header("Pages")
page = st.sidebar.radio(
    "", 
    ("Adaptive Test", "Item Bank", "How CAT Works"), 
    label_visibility="collapsed"
)

if page == "Adaptive Test":
    render_cat_page()
elif page == "Item Bank":
    render_bank_page()
else:                        
    render_explain_page()

st.sidebar.header("About")
st.sidebar.info(
    "This is a demo app for assessing the learning agility trait using Computerized Adaptive Testing (CAT). You can find the full source code on [GitHub](https://github.com/lstehlik2809/Computerized-Adaptive-Testing-Demo.git)."
)