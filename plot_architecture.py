"""
RDSFA-Net: Network Architecture & Module Diagrams
Generate publication-quality vector figures for paper.
Output: PDF/SVG files in figures/ directory
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# ============================================================
# Color Scheme
# ============================================================
COLORS = {
    'rgb': '#E74C3C',      # Red for RGB branch
    'ir': '#E67E22',        # Orange for IR branch
    'fusion': '#2ECC71',    # Green for fusion
    'head': '#3498DB',      # Blue for head
    'attn': '#9B59B6',      # Purple for attention
    'bg': '#F8F9FA',
    'text': '#2C3E50',
    'arrow': '#7F8C8D',
    'conv': '#85C1E9',
    'stem': '#F1948A',
    'align': '#F7DC6F',
    'degrad': '#F0B27A',
    'recovery': '#82E0AA',
}

def set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.edgecolor': '#2C3E50',
        'axes.linewidth': 0.8,
        'text.color': '#2C3E50',
    })

# ============================================================
# Figure 1: Overall Network Architecture
# ============================================================
def draw_overall_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.7, 'RDSFA-Net: Overall Architecture', 
            fontsize=14, fontweight='bold', ha='center', va='center',
            color=COLORS['text'])
    
    # ---- Input ----
    ax.text(1.5, 6.8, 'RGB+IR\nInput\n(6ch, 640x640)', fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', edgecolor='gray'))
    
    # ---- RGB Backbone ----
    # HGStem
    rect = FancyBboxPatch((3.5, 6.3), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor=COLORS['stem'], edgecolor=COLORS['rgb'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(4.5, 6.7, 'HGStem\n(3→48)', fontsize=7, ha='center', va='center')
    
    # Stage 1
    rect = FancyBboxPatch((3.5, 5.2), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FADBD8', edgecolor=COLORS['rgb'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(4.5, 5.6, 'ModalityHGBlock_RGB\nStage1 ×3 (48→128)', fontsize=6.5, ha='center', va='center')
    
    # Stage 2
    rect = FancyBboxPatch((3.5, 4.1), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FADBD8', edgecolor=COLORS['rgb'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(4.5, 4.5, 'ModalityHGBlock_RGB\nStage2 ×3 (96→256)', fontsize=6.5, ha='center', va='center')
    
    # Stage 3
    rect = FancyBboxPatch((3.5, 3.0), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#F5B7B1', edgecolor=COLORS['rgb'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(4.5, 3.4, 'ModalityHGBlock_RGB\nStage3 ×3×3 (192→768)', fontsize=6.5, ha='center', va='center')
    
    # Stage 4
    rect = FancyBboxPatch((3.5, 1.9), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#F5B7B1', edgecolor=COLORS['rgb'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(4.5, 2.3, 'ModalityHGBlock_RGB\nStage4 ×3 (384→1024)', fontsize=6.5, ha='center', va='center')
    
    # RGB branch label
    ax.text(4.5, 7.5, 'RGB Branch (SpatialAttention)', fontsize=8, fontweight='bold',
            ha='center', color=COLORS['rgb'])
    # Vertical line for RGB branch
    ax.plot([4.5, 4.5], [7.1, 6.3], color=COLORS['rgb'], linewidth=1, alpha=0.5)
    ax.plot([4.5, 4.5], [5.9, 5.2], color=COLORS['rgb'], linewidth=1, alpha=0.5)
    ax.plot([4.5, 4.5], [4.8, 4.1], color=COLORS['rgb'], linewidth=1, alpha=0.5)
    ax.plot([4.5, 4.5], [3.8, 3.0], color=COLORS['rgb'], linewidth=1, alpha=0.5)
    ax.plot([4.5, 4.5], [2.6, 1.9], color=COLORS['rgb'], linewidth=1, alpha=0.5)
    
    # ---- IR Backbone ----
    ax.text(9.5, 7.5, 'IR Branch (ChannelSEAttention)', fontsize=8, fontweight='bold',
            ha='center', color=COLORS['ir'])
    
    rect = FancyBboxPatch((8.5, 6.3), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FDEBD0', edgecolor=COLORS['ir'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.5, 6.7, 'HGStem\n(3→48)', fontsize=7, ha='center', va='center')
    
    rect = FancyBboxPatch((8.5, 5.2), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FDEBD0', edgecolor=COLORS['ir'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.5, 5.6, 'ModalityHGBlock_IR\nStage1 ×3 (48→128)', fontsize=6.5, ha='center', va='center')
    
    rect = FancyBboxPatch((8.5, 4.1), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FDEBD0', edgecolor=COLORS['ir'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.5, 4.5, 'ModalityHGBlock_IR\nStage2 ×3 (96→256)', fontsize=6.5, ha='center', va='center')
    
    rect = FancyBboxPatch((8.5, 3.0), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FAE5C3', edgecolor=COLORS['ir'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.5, 3.4, 'ModalityHGBlock_IR\nStage3 ×3×3 (192→768)', fontsize=6.5, ha='center', va='center')
    
    rect = FancyBboxPatch((8.5, 1.9), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FAE5C3', edgecolor=COLORS['ir'], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.5, 2.3, 'ModalityHGBlock_IR\nStage4 ×3 (384→1024)', fontsize=6.5, ha='center', va='center')
    
    # Vertical line for IR branch
    ax.plot([9.5, 9.5], [7.1, 6.3], color=COLORS['ir'], linewidth=1, alpha=0.5)
    ax.plot([9.5, 9.5], [5.9, 5.2], color=COLORS['ir'], linewidth=1, alpha=0.5)
    ax.plot([9.5, 9.5], [4.8, 4.1], color=COLORS['ir'], linewidth=1, alpha=0.5)
    ax.plot([9.5, 9.5], [3.8, 3.0], color=COLORS['ir'], linewidth=1, alpha=0.5)
    ax.plot([9.5, 9.5], [2.6, 1.9], color=COLORS['ir'], linewidth=1, alpha=0.5)
    
    # ---- Fusion Modules ----
    # RDRFA P5 (Stage4 outputs → fusion)
    rect = FancyBboxPatch((4.5, 1.1), 5.0, 0.6, boxstyle="round,pad=0.1",
                          facecolor=COLORS['fusion'], edgecolor='#1A7A3A', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.0, 1.4, 'RDRFA Fusion (P5)', fontsize=8, fontweight='bold', ha='center', va='center', color='white')
    # Arrows from both backbones to P5 fusion
    ax.annotate('', xy=(5.5, 1.7), xytext=(4.5, 1.9),
                arrowprops=dict(arrowstyle='->', color=COLORS['rgb'], lw=1.5))
    ax.annotate('', xy=(8.5, 1.7), xytext=(9.5, 1.9),
                arrowprops=dict(arrowstyle='->', color=COLORS['ir'], lw=1.5))
    
    # RDRFA P4 (Stage3 outputs → fusion)
    rect = FancyBboxPatch((4.5, 0.3), 5.0, 0.6, boxstyle="round,pad=0.1",
                          facecolor=COLORS['fusion'], edgecolor='#1A7A3A', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.0, 0.6, 'RDRFA Fusion (P4)', fontsize=8, fontweight='bold', ha='center', va='center', color='white')
    ax.annotate('', xy=(5.5, 0.9), xytext=(4.5, 3.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['rgb'], lw=1.5, linestyle='dashed'))
    ax.annotate('', xy=(8.5, 0.9), xytext=(9.5, 3.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['ir'], lw=1.5, linestyle='dashed'))
    
    # RDRFA P3 (Stage2 outputs → fusion)
    rect = FancyBboxPatch((4.5, -0.5), 5.0, 0.6, boxstyle="round,pad=0.1",
                          facecolor=COLORS['fusion'], edgecolor='#1A7A3A', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.0, -0.2, 'RDRFA Fusion (P3)', fontsize=8, fontweight='bold', ha='center', va='center', color='white')
    ax.annotate('', xy=(5.5, -0.5), xytext=(4.5, 4.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['rgb'], lw=1.5, linestyle='dashed'))
    ax.annotate('', xy=(8.5, -0.5), xytext=(9.5, 4.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['ir'], lw=1.5, linestyle='dashed'))
    
    # ---- Head ----
    # FPN + Decoder
    rect = FancyBboxPatch((2.0, -1.8), 10.0, 1.0, boxstyle="round,pad=0.15",
                          facecolor='#D4E6F1', edgecolor=COLORS['head'], linewidth=2)
    ax.add_patch(rect)
    ax.text(7.0, -1.3, 'Hybrid Encoder (AIFI) + RTDETR Decoder + FPN', 
            fontsize=9, fontweight='bold', ha='center', va='center', color=COLORS['head'])
    
    # Arrows from fusion to head
    for y_base, y_fusion in zip([-1.8, -1.8, -1.8], [-0.5, 0.3, 1.1]):
        ax.annotate('', xy=(4.0, y_base), xytext=(7.0, y_fusion),
                    arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.2))
    
    # Output
    rect = FancyBboxPatch((5.5, -2.9), 3.0, 0.7, boxstyle="round,pad=0.1",
                          facecolor='#1A5276', edgecolor=COLORS['head'], linewidth=2)
    ax.add_patch(rect)
    ax.text(7.0, -2.55, 'Detection Results\n(300 queries / 80 classes)', 
            fontsize=7, ha='center', va='center', color='white', fontweight='bold')
    ax.annotate('', xy=(7.0, -1.8), xytext=(7.0, -2.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['head'], lw=2))
    
    # Arrow from input to branches
    ax.annotate('', xy=(3.5, 6.7), xytext=(2.5, 6.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    ax.annotate('', xy=(8.5, 6.7), xytext=(2.5, 6.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5, connectionstyle='arc3,rad=0.3'))
    
    # Input arrow label
    ax.text(2.5, 7.0, 'Split', fontsize=7, ha='center', color=COLORS['arrow'])
    
    plt.tight_layout()
    fig.savefig('figures/fig1_overall_architecture.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('figures/fig1_overall_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('✅ Figure 1: Overall Architecture saved')


# ============================================================
# Figure 2: ModalityHGBlock Details
# ============================================================
def draw_modality_blocks():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # ---- Left: ModalityHGBlock_RGB ----
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 5.5)
    ax1.axis('off')
    ax1.text(3, 5.3, 'ModalityHGBlock_RGB', fontsize=11, fontweight='bold', ha='center', color=COLORS['rgb'])
    
    # Input
    ax1.text(3, 4.8, 'Input\n($C_{in}$)', fontsize=7, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))
    
    # HGBlock internal
    rect = FancyBboxPatch((1.0, 2.8), 4.0, 1.5, boxstyle="round,pad=0.1",
                          facecolor='#FADBD8', edgecolor=COLORS['rgb'], linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(3, 4.0, 'HGBlock Body', fontsize=8, fontweight='bold', ha='center', color=COLORS['rgb'])
    ax1.text(3, 3.7, 'n× LightConv/Conv + Squeeze + Excitation', fontsize=6.5, ha='center')
    ax1.text(3, 3.4, '($C_{in}$ → $C_{mid}$ → $C_{out}$)', fontsize=6.5, ha='center')
    ax1.text(3, 3.1, 'Shortcut connection (when $C_{in}=C_{out}$)', fontsize=6.5, ha='center', style='italic')
    
    # SpatialAttention
    rect = FancyBboxPatch((1.5, 1.2), 3.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor=COLORS['attn'], edgecolor='#6C3483', linewidth=1.5, alpha=0.85)
    ax1.add_patch(rect)
    ax1.text(3, 1.9, 'Spatial Attention', fontsize=8, fontweight='bold', ha='center', color='white')
    ax1.text(3, 1.6, 'AvgPool+MaxPool → Conv(3×3) → Sigmoid', fontsize=6.5, ha='center', color='white')
    
    # Arrows
    ax1.annotate('', xy=(3, 4.5), xytext=(3, 4.3),
                 arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.2))
    ax1.annotate('', xy=(3, 2.8), xytext=(3, 2.2),
                 arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.2))
    
    # Output
    ax1.text(3, 0.6, 'Output\n($C_{out}$)', fontsize=7, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#FADBD8', edgecolor=COLORS['rgb']))
    ax1.annotate('', xy=(3, 1.2), xytext=(3, 0.8),
                 arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.2))
    
    # ---- Right: ModalityHGBlock_IR ----
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 5.5)
    ax2.axis('off')
    ax2.text(3, 5.3, 'ModalityHGBlock_IR', fontsize=11, fontweight='bold', ha='center', color=COLORS['ir'])
    
    # Input
    ax2.text(3, 4.8, 'Input\n($C_{in}$)', fontsize=7, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))
    
    # HGBlock internal (same)
    rect = FancyBboxPatch((1.0, 2.8), 4.0, 1.5, boxstyle="round,pad=0.1",
                          facecolor='#FDEBD0', edgecolor=COLORS['ir'], linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(3, 4.0, 'HGBlock Body', fontsize=8, fontweight='bold', ha='center', color=COLORS['ir'])
    ax2.text(3, 3.7, 'n× LightConv/Conv + Squeeze + Excitation', fontsize=6.5, ha='center')
    ax2.text(3, 3.4, '($C_{in}$ → $C_{mid}$ → $C_{out}$)', fontsize=6.5, ha='center')
    ax2.text(3, 3.1, 'Shortcut connection (when $C_{in}=C_{out}$)', fontsize=6.5, ha='center', style='italic')
    
    # ChannelSEAttention
    rect = FancyBboxPatch((1.5, 1.2), 3.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#D35400', edgecolor='#935116', linewidth=1.5, alpha=0.85)
    ax2.add_patch(rect)
    ax2.text(3, 1.9, 'Channel SE Attention', fontsize=8, fontweight='bold', ha='center', color='white')
    ax2.text(3, 1.6, 'GAP → Conv(1×1) → ReLU → Conv(1×1) → Sigmoid', fontsize=6.5, ha='center', color='white')
    
    # Arrows
    ax2.annotate('', xy=(3, 4.5), xytext=(3, 4.3),
                 arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.2))
    ax2.annotate('', xy=(3, 2.8), xytext=(3, 2.2),
                 arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.2))
    ax2.annotate('', xy=(3, 1.2), xytext=(3, 0.8),
                 arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.2))
    
    # Output
    ax2.text(3, 0.6, 'Output\n($C_{out}$)', fontsize=7, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEBD0', edgecolor=COLORS['ir']))
    
    plt.tight_layout()
    fig.savefig('figures/fig2_modality_blocks.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('figures/fig2_modality_blocks.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('✅ Figure 2: ModalityHGBlocks saved')


# ============================================================
# Figure 3: RDRFusion Module Detail
# ============================================================
def draw_rdrfusion():
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    ax.text(7, 6.8, 'RDRFA Fusion Module (Recovery-Driven Degradation-aware Fusion with Alignment)',
            fontsize=12, fontweight='bold', ha='center', color=COLORS['fusion'])
    
    # ---- Inputs ----
    ax.text(2.5, 6.2, '$F_{rgb}$\n(RGB features)', fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor=COLORS['rgb'], linewidth=1.5))
    ax.text(11.5, 6.2, '$F_{ir}$\n(IR features)', fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', edgecolor=COLORS['ir'], linewidth=1.5))
    
    # ---- Step 1: Alignment ----
    rect = FancyBboxPatch((3.0, 4.5), 8.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor=COLORS['align'], edgecolor='#B7950B', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7, 5.4, '① DeformableOffsetAlignment', fontsize=9, fontweight='bold', ha='center')
    ax.text(7, 5.0, 'Conv(3×3) → Tanh → Offset Prediction → Grid Sample (bi-directional alignment)',
            fontsize=7, ha='center', color='#7D6608')
    
    # Arrows in
    ax.annotate('', xy=(3.0, 5.7), xytext=(2.5, 6.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['rgb'], lw=1.5))
    ax.annotate('', xy=(11.0, 5.7), xytext=(11.5, 6.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['ir'], lw=1.5))
    
    # ---- Step 2: Degradation Estimator ----
    rect = FancyBboxPatch((3.0, 3.0), 8.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor=COLORS['degrad'], edgecolor='#CA6F1E', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7, 3.9, '② DegradationEstimator', fontsize=9, fontweight='bold', ha='center')
    ax.text(7, 3.55, 'DWConv(3×3) → DWConv(3×3) → Conv(3×3) → Sigmoid', fontsize=7, ha='center', color='#7E5109')
    ax.text(7, 3.2, '$D_{rgb}$  $D_{ir}$  $D_{blur}$  $R_{rgb}$  $R_{ir}$', fontsize=8, ha='center',
            fontweight='bold', color='#A04000')
    
    # Arrow
    ax.annotate('', xy=(7, 4.5), xytext=(7, 4.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    
    # ---- Step 3: Recovery ----
    # Recovery labels
    ax.text(1.2, 2.5, '$F_{rgb}^{clean} = F_{rgb}^{align} \\times R_{rgb}$', fontsize=7,
            fontweight='bold', color=COLORS['rgb'])
    ax.text(12.8, 2.5, '$F_{ir}^{clean} = F_{ir}^{align} \\times R_{ir}$', fontsize=7,
            fontweight='bold', color=COLORS['ir'])
    
    # ---- Step 4: Fusion ----
    rect = FancyBboxPatch((3.0, 0.5), 8.0, 1.5, boxstyle="round,pad=0.1",
                          facecolor=COLORS['recovery'], edgecolor='#1E8449', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7, 1.8, '③ RecoveryGuidedFusion', fontsize=9, fontweight='bold', ha='center')
    ax.text(7, 1.5, 'Shared-private Feature Decomposition', fontsize=7.5, ha='center')
    ax.text(7, 1.2, '$F_{shared} = SharedConv(Mix)$  |  $F_{priv}^{rgb} = PrivConv(F_{rgb}^{clean})$  |  $F_{priv}^{ir} = PrivConv(F_{ir}^{clean})$', 
            fontsize=6.5, ha='center')
    ax.text(7, 0.9, '$F_{fusion} = Proj(Concat[F_{shared}, F_{priv}^{rgb}, F_{priv}^{ir}])$',
            fontsize=7, ha='center', fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(7, 3.0), xytext=(7, 2.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    
    # ---- Output ----
    ax.text(7, -0.2, '$F_{fusion}$  (C channels)', fontsize=9, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['fusion'], edgecolor='#1A7A3A',
                      linewidth=2, alpha=0.85))
    ax.annotate('', xy=(7, 0.5), xytext=(7, 0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['fusion'], lw=2))
    
    plt.tight_layout()
    fig.savefig('figures/fig3_rdrfusion_module.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('figures/fig3_rdrfusion_module.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('✅ Figure 3: RDRFusion Module saved')


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    set_style()
    draw_overall_architecture()
    draw_modality_blocks()
    draw_rdrfusion()
    print('\n🎉 All figures saved to figures/ directory')
    print('   - figures/fig1_overall_architecture.pdf')
    print('   - figures/fig2_modality_blocks.pdf')
    print('   - figures/fig3_rdrfusion_module.pdf')
