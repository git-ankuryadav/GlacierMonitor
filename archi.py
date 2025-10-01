
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_architecture_diagram():
    """Create a visual diagram of the Attention DeepLab V3+ architecture"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Define colors
    colors = {
        'input': '#E8F4FD',
        'encoder': '#B8E6B8', 
        'aspp': '#FFD700',
        'attention': '#FF6B6B',
        'decoder': '#DDA0DD',
        'output': '#98FB98'
    }

    # Input
    input_box = FancyBboxPatch((0.5, 8), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 8.5, 'Input\n(512Ã—512Ã—5)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Encoder stages
    encoder_stages = [
        ('Conv1+BN+ReLU', 4, 7.5),
        ('MaxPool+Layer1', 4, 6.5),
        ('Layer2', 4, 5.5),
        ('Layer3', 4, 4.5),
        ('Layer4', 4, 3.5)
    ]

    for i, (name, x, y) in enumerate(encoder_stages):
        box = FancyBboxPatch((x, y), 2.5, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['encoder'],
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 1.25, y + 0.4, name, ha='center', va='center', fontsize=9)

    # ASPP with CBAM
    aspp_box = FancyBboxPatch((8, 3), 4, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['aspp'],
                             edgecolor='black', linewidth=2)
    ax.add_patch(aspp_box)
    ax.text(10, 4, 'Attention ASPP\n+ CBAM\n(Multiple Scales)', 
            ha='center', va='center', fontsize=10, fontweight='bold')

    # CBAM modules
    cbam_box = FancyBboxPatch((8.5, 1.5), 3, 1,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['attention'],
                             edgecolor='black', linewidth=1)
    ax.add_patch(cbam_box)
    ax.text(10, 2, 'Channel + Spatial\nAttention', 
            ha='center', va='center', fontsize=9)

    # Low-level projection
    low_proj_box = FancyBboxPatch((4, 2), 2.5, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors['decoder'],
                                 edgecolor='black', linewidth=1)
    ax.add_patch(low_proj_box)
    ax.text(5.25, 2.4, 'Low-level\nProjection', ha='center', va='center', fontsize=9)

    # Decoder
    decoder_box = FancyBboxPatch((13, 2), 3, 2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['decoder'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(14.5, 3, 'Decoder\n(Depthwise\nSeparable Conv)', 
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Output
    output_box = FancyBboxPatch((13.5, 0.5), 2, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['output'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(14.5, 1, 'Binary Mask\n(512Ã—512Ã—1)', ha='center', va='center', 
            fontsize=10, fontweight='bold')

    # Arrows
    arrows = [
        # Main flow
        ((1.5, 8), (5.25, 7.8)),  # Input to encoder
        ((5.25, 7.1), (5.25, 6.9)),  # Encoder flow
        ((5.25, 6.1), (5.25, 5.9)),
        ((5.25, 5.1), (5.25, 4.9)),
        ((5.25, 4.1), (5.25, 3.9)),
        ((6.5, 3.9), (8, 4)),  # To ASPP
        ((10, 3), (10, 2.5)),  # ASPP to CBAM
        ((12, 4), (13, 3.5)),  # ASPP to decoder

        # Skip connection
        ((5.25, 6.5), (5.25, 2.8)),  # Low-level features
        ((6.5, 2.4), (13, 2.8)),  # To decoder

        # Final output
        ((14.5, 2), (14.5, 1.5))  # Decoder to output
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    # Add TTA annotation
    tta_box = FancyBboxPatch((0.5, 0.5), 3, 1,
                            boxstyle="round,pad=0.1",
                            facecolor='#FFA07A',
                            edgecolor='black', linewidth=1)
    ax.add_patch(tta_box)
    ax.text(2, 1, 'Test-Time\nAugmentation\n(4 transforms)', 
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['input'], label='Input Layer'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['encoder'], label='Encoder (ResNet-34)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['aspp'], label='ASPP Module'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['attention'], label='CBAM Attention'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['decoder'], label='Decoder'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['output'], label='Output')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.title('Attention DeepLab V3+ Architecture for Glacier Segmentation', 
              fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()