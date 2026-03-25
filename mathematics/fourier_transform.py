import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# --- Configuration ---
# Signal s(t) — change this to experiment
t = np.linspace(-2, 2, 2000)
s = np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * np.pi * 7 * t)

f_min, f_max, f_step = 0.1, 15.0, 0.1
freqs_slider = np.arange(f_min, f_max + f_step / 2, f_step)

# --- Palette: Dieter Rams — warm light grey, orange accent, restrained ---
BG = '#e8e4df'
SURFACE = '#f2efeb'
BORDER = '#ccc8c2'
TEXT = '#2c2a27'
TEXT_DIM = '#8a8680'
ORANGE = '#d45a1a'
ORANGE_SOFT = 'rgba(212,90,26,0.10)'
BLUE = '#3068a8'
BLUE_SOFT = 'rgba(48,104,168,0.08)'
TEAL = '#1a7a6a'
TEAL_SOFT = 'rgba(26,122,106,0.07)'

MONO = '"JetBrains Mono", monospace'
SERIF = '"Source Serif 4", Georgia, serif'

# --- Precompute |S(f)| for slider colorbar ---
spectrum_mag = np.array([
    abs(np.trapezoid(s * np.exp(-2j * np.pi * fr * t), t))
    for fr in freqs_slider
])
mag_max = spectrum_mag.max() if spectrum_mag.max() > 0 else 1.0


def mag_to_color(mag):
    n = mag / mag_max
    r = int(200 + n * (212 - 200))
    g = int(196 + n * (90 - 196))
    b = int(190 + n * (26 - 190))
    return f'rgb({r},{g},{b})'


def build_gradient_css():
    stops = []
    for i, fr in enumerate(freqs_slider):
        pct = (fr - f_min) / (f_max - f_min) * 100
        stops.append(f'{mag_to_color(spectrum_mag[i])} {pct:.1f}%')
    return f'linear-gradient(90deg, {", ".join(stops)})'


gradient = build_gradient_css()

app = Dash(__name__)

app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,500;0,8..60,700;1,8..60,400&family=JetBrains+Mono:wght@400;600&display=swap'
    ),

    # Header
    html.Div([
        html.H1('Fourier Transform', style={
            'margin': '0', 'fontSize': '1.5rem', 'fontWeight': '300',
            'fontFamily': SERIF, 'color': TEXT,
            'letterSpacing': '0.03em',
        }),
        html.Div(style={
            'width': '24px', 'height': '2px', 'background': ORANGE,
            'marginTop': '6px',
        }),
        html.P(
            'S(f) = ∫ s(t) · e⁻²ʲᵖᶠᵗ dt',
            style={
                'margin': '8px 0 0', 'fontSize': '0.8rem',
                'fontFamily': MONO, 'color': TEXT_DIM,
            }
        ),
    ], style={'padding': '28px 48px 0'}),

    # Result readout
    html.Div(id='result-readout', style={
        'padding': '14px 48px 0',
        'fontFamily': MONO, 'fontSize': '0.8rem',
    }),

    # Slider
    html.Div([
        html.Div([
            html.Span('f', style={
                'fontFamily': SERIF, 'fontStyle': 'italic',
                'fontSize': '0.95rem', 'color': TEXT,
            }),
            html.Span('  Hz', style={
                'fontFamily': MONO, 'fontSize': '0.65rem',
                'color': TEXT_DIM, 'marginLeft': '3px',
            }),
        ], style={'marginBottom': '6px'}),
        dcc.Slider(
            id='freq-slider',
            min=f_min, max=f_max, step=f_step,
            value=3.0,
            marks={i: {'label': str(i), 'style': {
                'fontFamily': MONO, 'fontSize': '0.6rem', 'color': TEXT_DIM,
            }} for i in range(1, int(f_max) + 1, 2)},
            tooltip={'always_visible': False},
            className='spectrum-slider',
        ),
        html.Div([
            html.Span('0', style={'color': TEXT_DIM, 'fontSize': '0.6rem'}),
            html.Span('|S(f)|', style={'color': TEXT_DIM, 'fontSize': '0.6rem'}),
            html.Span('max', style={'color': ORANGE, 'fontSize': '0.6rem'}),
        ], style={
            'display': 'flex', 'justifyContent': 'space-between',
            'fontFamily': MONO, 'padding': '1px 8px 0',
        }),
    ], style={'padding': '8px 48px 0'}),

    # Plot
    dcc.Graph(id='plot', style={'height': '66vh'},
              config={'displayModeBar': 'hover', 'displaylogo': False}),

], style={'background': BG, 'minHeight': '100vh'})

app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Fourier Transform</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; background: ''' + BG + '''; }

        /* Kill any white borders/outlines on the Dash graph container */
        .dash-graph, .js-plotly-plot, .plot-container,
        .main-svg, .svg-container {
            background: transparent !important;
            border: none !important;
            outline: none !important;
        }

        /* Slider — override every possible selector Dash uses */
        .spectrum-slider .rc-slider,
        .spectrum-slider [class*="slider"] {
            background: transparent !important;
        }
        .spectrum-slider .rc-slider-track,
        .spectrum-slider [class*="track"] {
            background: transparent !important;
            border: none !important;
        }
        .spectrum-slider .rc-slider-rail,
        .spectrum-slider [class*="rail"] {
            background: ''' + gradient + ''' !important;
            height: 6px !important;
            border-radius: 3px !important;
            opacity: 1 !important;
            border: none !important;
        }
        .spectrum-slider .rc-slider-handle,
        .spectrum-slider [class*="handle"],
        .spectrum-slider [role="slider"] {
            width: 14px !important;
            height: 14px !important;
            margin-top: -5px !important;
            border: 2px solid ''' + TEXT + ''' !important;
            background: ''' + SURFACE + ''' !important;
            box-shadow: none !important;
            opacity: 1 !important;
            outline: none !important;
        }
        .spectrum-slider .rc-slider-handle:hover,
        .spectrum-slider .rc-slider-handle:active,
        .spectrum-slider .rc-slider-handle:focus,
        .spectrum-slider .rc-slider-handle-dragging,
        .spectrum-slider [class*="handle"]:hover,
        .spectrum-slider [class*="handle"]:active,
        .spectrum-slider [class*="handle"]:focus,
        .spectrum-slider [role="slider"]:hover,
        .spectrum-slider [role="slider"]:active,
        .spectrum-slider [role="slider"]:focus {
            border-color: ''' + ORANGE + ''' !important;
            box-shadow: none !important;
            outline: none !important;
        }
        .spectrum-slider .rc-slider-step,
        .spectrum-slider [class*="step"] {
            height: 6px !important;
        }
        .spectrum-slider .rc-slider-dot,
        .spectrum-slider [class*="dot"] {
            display: none !important;
        }

        /* Dash tooltip — hide it completely */
        .spectrum-slider [class*="tooltip"],
        .rc-slider-tooltip {
            display: none !important;
        }

        /* Plotly modebar on theme */
        .modebar-btn path { fill: ''' + TEXT_DIM + ''' !important; }
        .modebar-btn:hover path { fill: ''' + ORANGE + ''' !important; }
        .modebar { background: transparent !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''


@app.callback(
    Output('plot', 'figure'),
    Output('result-readout', 'children'),
    Input('freq-slider', 'value'),
)
def update(f):
    kernel = np.exp(-2j * np.pi * f * t)
    product = s * kernel
    S_f = np.trapezoid(product, t)

    half = 0.55 / f
    t_lo, t_hi = -half, half

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=s, name='s(t)',
        line=dict(color=ORANGE, width=2),
        hovertemplate='t=%{x:.4f}  s=%{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=t, y=kernel.real, name='Re{ e⁻²ʲᵖᶠᵗ }',
        line=dict(color=BLUE, width=1.5),
        hovertemplate='t=%{x:.4f}  cos=%{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=t, y=kernel.imag, name='Im{ e⁻²ʲᵖᶠᵗ }',
        line=dict(color=BLUE, width=1.2, dash='dot'), opacity=0.4,
        hovertemplate='t=%{x:.4f}  sin=%{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=t, y=product.real, name='Re{ s · e⁻²ʲᵖᶠᵗ }',
        line=dict(color=TEAL, width=1.8),
        fill='tozeroy', fillcolor=TEAL_SOFT,
        hovertemplate='t=%{x:.4f}  Re=%{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=t, y=product.imag, name='Im{ s · e⁻²ʲᵖᶠᵗ }',
        line=dict(color=TEAL, width=1.2, dash='dot'), opacity=0.35,
        fill='tozeroy', fillcolor='rgba(26,122,106,0.03)',
        hovertemplate='t=%{x:.4f}  Im=%{y:.4f}<extra></extra>',
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text='t (s)', font=dict(family=MONO, size=11, color=TEXT_DIM)),
            range=[t_lo, t_hi],
            gridcolor=BORDER, gridwidth=1,
            zeroline=True, zerolinecolor='#b0aca6', zerolinewidth=1,
            tickfont=dict(family=MONO, size=10, color=TEXT_DIM),
            linecolor=BORDER,
        ),
        yaxis=dict(
            title=dict(text='amplitude', font=dict(family=MONO, size=11, color=TEXT_DIM)),
            gridcolor=BORDER, gridwidth=1,
            zeroline=True, zerolinecolor='#b0aca6', zerolinewidth=1,
            tickfont=dict(family=MONO, size=10, color=TEXT_DIM),
            linecolor=BORDER,
        ),
        plot_bgcolor=SURFACE,
        paper_bgcolor=BG,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(family=MONO, size=11, color=TEXT),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(t=24, b=44, l=56, r=24),
        hoverlabel=dict(
            bgcolor=SURFACE, font_color=TEXT,
            font=dict(family=MONO, size=11),
            bordercolor=BORDER,
        ),
    )

    sep = html.Span('  ·  ', style={'color': BORDER})
    readout = html.Div([
        html.Span('S', style={
            'fontFamily': SERIF, 'fontStyle': 'italic', 'color': TEXT,
        }),
        html.Span(f'({f:.1f})', style={'color': TEXT_DIM}),
        html.Span(' = ', style={'color': BORDER}),
        html.Span(f'{S_f.real:+.4f}{S_f.imag:+.4f}j', style={
            'color': ORANGE, 'fontWeight': '600',
        }),
        sep,
        html.Span('|S| = ', style={'color': TEXT_DIM}),
        html.Span(f'{abs(S_f):.4f}', style={'color': TEXT, 'fontWeight': '600'}),
        sep,
        html.Span('∠ ', style={'color': TEXT_DIM}),
        html.Span(f'{np.degrees(np.angle(S_f)):.1f}°', style={
            'color': TEXT, 'fontWeight': '600',
        }),
    ])

    return fig, readout


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
