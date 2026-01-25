import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
from src.maintenance_engine import maintenance_engine

def analyze_asset(asset_id, degradation_rate):
    # Simulate history
    cycles = 50
    health = 100 - np.arange(cycles) * (degradation_rate / 10) + np.random.normal(0, 1, cycles)
    history = [{"cycle": i, "health_index": h} for i, h in enumerate(health)]
    
    rul = maintenance_engine.predict_rul(history)
    strategy = maintenance_engine.get_maintenance_strategy(asset_id, rul)
    
    df = pd.DataFrame(history)
    fig = px.line(df, x="cycle", y="health_index", title=f"Health Degradation - {asset_id}")
    fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    
    return fig, f"### Predicted RUL: {rul} Cycles\n\n{strategy}"

# ============================================
# GRADIO UI
# ============================================

with gr.Blocks(title="Predictive Maintenance", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Predictive Maintenance
    ### AI-Driven Grid Reliability
    
    Predicting Remaining Useful Life (RUL) and generating automated procurement strategies.
    """)
    
    with gr.Row():
        with gr.Column():
            asset_input = gr.Textbox(label="Asset ID", value="CNG-COMP-001")
            degrad_slider = gr.Slider(minimum=1, maximum=20, value=5, label="Observed Degradation Rate")
            predict_btn = gr.Button("Predict Failure", variant="primary")
            
        with gr.Column():
            plot_output = gr.Plot(label="Health Index Trend")
            
    strategy_output = gr.Markdown(label="Maintenance Strategy")
    
    predict_btn.click(
        fn=analyze_asset,
        inputs=[asset_input, degrad_slider],
        outputs=[plot_output, strategy_output]
    )
    
    gr.Markdown("""
    ---
    **Tech Stack:** Scikit-Learn • Mistral-7B • Plotly • Gradio
    
    **Author:** [David Fernandez](https://davidfernandez.dev) | AI Engineer
    """)

if __name__ == "__main__":
    demo.launch()
