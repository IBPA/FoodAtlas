import plotly.graph_objects as go

fig = go.Figure()

x = [0.1, 0.3, 0.5, 0.7, 0.9]

fig.add_trace(go.Scatter(
    x=x,
    y=[0.9, 0.7, 0.5, 0.7, 0.9],
    name='Hypothetical',
    connectgaps=True,
))

fig.add_trace(go.Scatter(
    x=x,
    y=[0.20, 0.063, 0.16, 0.611, 0.476],
    name='Actual',
    connectgaps=True,
))

fig.update_xaxes(tickmode="linear")
fig.write_image("../../outputs/analysis_codes/all_literature_gt_calibration_plot.svg")
fig.write_image("../../outputs/analysis_codes/all_literature_gt_calibration_plot.png")
