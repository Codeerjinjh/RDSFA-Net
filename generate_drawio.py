"""
Generate professional .drawio file for RDSFA-Net architecture diagrams.
Open with https://app.diagrams.net
"""
import os
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

os.makedirs('figures', exist_ok=True)

def make_style(**kwargs):
    """Convert style dict to drawio style string."""
    return ';'.join(f'{k}={v}' for k, v in kwargs.items())

def cell(attrs, content=''):
    """Create a drawio cell element."""
    el = ET.Element('mxCell', attrs)
    if content:
        el.text = content
    return el

def geometry(x, y, w, h):
    el = ET.Element('mxGeometry', {'x': str(x), 'y': str(y), 'width': str(w), 'height': str(h), 'as': 'geometry'})
    return el

# ============================================================
# Page 1: Overall Architecture
# ============================================================
def build_architecture_page():
    root = ET.Element('mxGraphModel', {'dx': '0', 'dy': '0', 'grid': '1', 'gridSize': '10',
                                        'guides': '1', 'tooltips': '1', 'connect': '1',
                                        'arrows': '1', 'fold': '1', 'page': '1', 'pageScale': '1',
                                        'pageWidth': '1200', 'pageHeight': '900'})
    root_root = ET.SubElement(root, 'root')
    # Parent cell
    ET.SubElement(root_root, 'mxCell', {'id': '0'})
    ET.SubElement(root_root, 'mxCell', {'id': '1', 'parent': '0'})
    
    cells = []
    cid = [2]
    
    def add_cell(content, style, x, y, w, h, parent='1'):
        cid[0] += 1
        attrs = {'id': str(cid[0]), 'value': escape(content), 'style': style,
                 'vertex': '1', 'parent': parent}
        if x is not None:
            attrs['x'] = str(x)
            attrs['y'] = str(y)
            attrs['width'] = str(w)
            attrs['height'] = str(h)
        el = ET.SubElement(root_root, 'mxCell', attrs)
        if x is not None:
            geo = ET.SubElement(el, 'mxGeometry', {'width': str(w), 'height': str(h), 'as': 'geometry'})
        return cid[0]
    
    def add_arrow(src, dst, style='', label='', entry='', exit=''):
        cid[0] += 1
        attrs = {'id': str(cid[0]), 'style': f'edgeStyle=orthogonalEdgeStyle;rounded=0;{style}',
                 'edge': '1', 'parent': '1', 'source': str(src), 'target': str(dst)}
        el = ET.SubElement(root_root, 'mxCell', attrs)
        if label:
            ET.SubElement(el, 'mxCell', {'id': str(cid[0])+'_l', 'value': escape(label),
                                         'style': 'text;html=1;align=center;verticalAlign=middle;fontSize=8;',
                                         'vertex': '1', 'connectable': '0', 'parent': str(cid[0])})
        geo = ET.SubElement(el, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
        if entry:
            arr_pt = ET.SubElement(geo, 'Array', {'as': 'points'})
            ET.SubElement(arr_pt, 'mxPoint', {'x': entry, 'y': exit})
        return cid[0]
    
    # Title
    add_cell('RDSFA-Net: Overall Architecture',
             'text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;',
             330, 10, 540, 40)
    
    # ---- Input ----
    add_cell('RGB+IR Input&lt;br&gt;6×640×640',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#666666;fontSize=9;',
             500, 60, 120, 50)
    
    # ---- RGB Branch ----
    # Branch label
    add_cell('RGB Branch (Spatial Attention)',
             'text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;color=#C0392B;',
             200, 120, 160, 30)
    
    # Stem
    id_rgb_stem = add_cell('HGStem<br>3 → 48',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#FADBD8;strokeColor=#C0392B;fontSize=8;',
             180, 160, 120, 50)
    
    # Stage labels
    add_cell('ModalityHGBlock_RGB<br>Stage 1 ×3  (48→128)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#FADBD8;strokeColor=#C0392B;fontSize=8;',
             180, 230, 120, 50)
    add_cell('ModalityHGBlock_RGB<br>Stage 2 ×3  (96→256)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#FADBD8;strokeColor=#C0392B;fontSize=8;',
             180, 300, 120, 50)
    id_rgb_s3 = add_cell('ModalityHGBlock_RGB<br>Stage 3 ×3×3 (192→768)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#F1948A;strokeColor=#C0392B;fontSize=8;',
             180, 370, 120, 50)
    id_rgb_s4 = add_cell('ModalityHGBlock_RGB<br>Stage 4 ×3 (384→1024)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#F1948A;strokeColor=#C0392B;fontSize=8;',
             180, 440, 120, 50)
    
    # ---- IR Branch ----
    id_ir_stem = add_cell('IR Branch (Channel SE Attention)',
             'text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;color=#D35400;',
             740, 120, 140, 30)
    
    add_cell('HGStem<br>3 → 48',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#FDEBD0;strokeColor=#D35400;fontSize=8;',
             700, 160, 120, 50)
    add_cell('ModalityHGBlock_IR<br>Stage 1 ×3  (48→128)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#FDEBD0;strokeColor=#D35400;fontSize=8;',
             700, 230, 120, 50)
    add_cell('ModalityHGBlock_IR<br>Stage 2 ×3  (96→256)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#FDEBD0;strokeColor=#D35400;fontSize=8;',
             700, 300, 120, 50)
    id_ir_s3 = add_cell('ModalityHGBlock_IR<br>Stage 3 ×3×3 (192→768)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#FAE5C3;strokeColor=#D35400;fontSize=8;',
             700, 370, 120, 50)
    id_ir_s4 = add_cell('ModalityHGBlock_IR<br>Stage 4 ×3 (384→1024)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#FAE5C3;strokeColor=#D35400;fontSize=8;',
             700, 440, 120, 50)
    
    # ---- Fusion Modules ----
    id_fuse_p5 = add_cell('RDRFA Fusion (P5)<br>F = RDRFusion(F_rgb, F_ir)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#2ECC71;strokeColor=#1A7A3A;fontSize=9;fontStyle=1;fontColor=#FFFFFF;',
             370, 510, 260, 50)
    id_fuse_p4 = add_cell('RDRFA Fusion (P4)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#2ECC71;strokeColor=#1A7A3A;fontSize=9;fontStyle=1;fontColor=#FFFFFF;',
             370, 580, 260, 50)
    id_fuse_p3 = add_cell('RDRFA Fusion (P3)',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#2ECC71;strokeColor=#1A7A3A;fontSize=9;fontStyle=1;fontColor=#FFFFFF;',
             370, 650, 260, 50)
    
    # ---- Head ----
    id_head = add_cell('Hybrid Encoder (AIFI) + RTDETR Decoder + FPN',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#D4E6F1;strokeColor=#2471A3;fontSize=11;fontStyle=1;',
             270, 730, 460, 50)
    id_out = add_cell('Detection Output<br>300 queries / NMS',
             'rounded=1;whiteSpace=wrap;html=1;fillColor=#1A5276;strokeColor=#2471A3;fontSize=9;fontStyle=1;fontColor=#FFFFFF;',
             370, 810, 260, 40)
    
    # Arrows from input to branches
    # We'll use simple connector edges via source/target
    # For orthogonal edges we need to set the edge style
    
    return root

# Actually, let me use a simpler approach - generate SVG file instead,
# which is more reliable and easier to edit in draw.io

def create_architecture_svg():
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="1100" height="900" viewBox="0 0 1100 900">
  <defs>
    <marker id="arrowGray" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#7F8C8D"/>
    </marker>
    <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#E74C3C"/>
    </marker>
    <marker id="arrowOrange" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#E67E22"/>
    </marker>
    <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#27AE60"/>
    </marker>
    <marker id="arrowBlue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#2980B9"/>
    </marker>
  </defs>

  <style>
    .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #2C3E50; }
    .label { font-family: Arial, sans-serif; font-size: 11px; fill: #2C3E50; }
    .small { font-family: Arial, sans-serif; font-size: 9px; fill: #2C3E50; }
    .white { font-family: Arial, sans-serif; font-size: 9px; fill: white; }
    .box { fill: #F8F9FA; stroke: #BDC3C7; stroke-width: 1.5; rx: 6; ry: 6; }
  </style>

  <!-- Title -->
  <text x="550" y="30" text-anchor="middle" class="title">RDSFA-Net: Overall Architecture</text>

  <!-- Input -->
  <rect x="470" y="55" width="160" height="45" rx="8" ry="8" fill="#F0F0F0" stroke="#7F8C8D" stroke-width="1.5"/>
  <text x="550" y="73" text-anchor="middle" class="label" font-weight="bold">RGB + IR Input</text>
  <text x="550" y="90" text-anchor="middle" class="small">6 channels, 640×640</text>

  <!-- Split arrow -->
  <line x1="550" y1="100" x2="550" y2="130" stroke="#7F8C8D" stroke-width="1.5" marker-end="url(#arrowGray)"/>
  <line x1="550" y1="115" x2="240" y2="115" stroke="#7F8C8D" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="550" y1="115" x2="860" y2="115" stroke="#7F8C8D" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="550" y="112" text-anchor="middle" class="small" fill="#7F8C8D">Channel Split</text>

  <!-- RGB Branch Label -->
  <text x="240" y="150" text-anchor="middle" class="label" font-weight="bold" fill="#E74C3C">RGB Branch (SpatialAttention)</text>
  <rect x="170" y="180" width="140" height="45" rx="6" ry="6" fill="#FADBD8" stroke="#E74C3C" stroke-width="1.5"/>
  <text x="240" y="200" text-anchor="middle" class="small" font-weight="bold">HGStem</text>
  <text x="240" y="215" text-anchor="middle" class="small">3 → 48, stride 4</text>

  <rect x="170" y="250" width="140" height="45" rx="6" ry="6" fill="#F5B7B1" stroke="#E74C3C" stroke-width="1.5"/>
  <text x="240" y="267" text-anchor="middle" class="small" font-weight="bold">ModalityHGBlock_RGB</text>
  <text x="240" y="282" text-anchor="middle" class="small">Stage 1 ×3 (48 → 128)</text>

  <rect x="170" y="320" width="140" height="45" rx="6" ry="6" fill="#F5B7B1" stroke="#E74C3C" stroke-width="1.5"/>
  <text x="240" y="337" text-anchor="middle" class="small" font-weight="bold">ModalityHGBlock_RGB</text>
  <text x="240" y="352" text-anchor="middle" class="small">Stage 2 ×3 (96 → 256)</text>

  <rect x="170" y="390" width="140" height="45" rx="6" ry="6" fill="#E74C3C" stroke="#C0392B" stroke-width="1.5" opacity="0.8"/>
  <text x="240" y="407" text-anchor="middle" class="white" font-weight="bold">ModalityHGBlock_RGB</text>
  <text x="240" y="422" text-anchor="middle" class="white">Stage 3 ×3×3 (192 → 768)</text>

  <rect x="170" y="460" width="140" height="45" rx="6" ry="6" fill="#E74C3C" stroke="#C0392B" stroke-width="1.5" opacity="0.8"/>
  <text x="240" y="477" text-anchor="middle" class="white" font-weight="bold">ModalityHGBlock_RGB</text>
  <text x="240" y="492" text-anchor="middle" class="white">Stage 4 ×3 (384 → 1024)</text>

  <!-- RGB branch arrows -->
  <line x1="240" y1="225" x2="240" y2="250" stroke="#E74C3C" stroke-width="1" marker-end="url(#arrowRed)"/>
  <line x1="240" y1="295" x2="240" y2="320" stroke="#E74C3C" stroke-width="1" marker-end="url(#arrowRed)"/>
  <line x1="240" y1="365" x2="240" y2="390" stroke="#E74C3C" stroke-width="1" marker-end="url(#arrowRed)"/>
  <line x1="240" y1="435" x2="240" y2="460" stroke="#E74C3C" stroke-width="1" marker-end="url(#arrowRed)"/>

  <!-- IR Branch Label -->
  <text x="860" y="150" text-anchor="middle" class="label" font-weight="bold" fill="#D35400">IR Branch (ChannelSEAttention)</text>
  <rect x="790" y="180" width="140" height="45" rx="6" ry="6" fill="#FDEBD0" stroke="#D35400" stroke-width="1.5"/>
  <text x="860" y="200" text-anchor="middle" class="small" font-weight="bold">HGStem</text>
  <text x="860" y="215" text-anchor="middle" class="small">3 → 48, stride 4</text>

  <rect x="790" y="250" width="140" height="45" rx="6" ry="6" fill="#FAE5C3" stroke="#D35400" stroke-width="1.5"/>
  <text x="860" y="267" text-anchor="middle" class="small" font-weight="bold">ModalityHGBlock_IR</text>
  <text x="860" y="282" text-anchor="middle" class="small">Stage 1 ×3 (48 → 128)</text>

  <rect x="790" y="320" width="140" height="45" rx="6" ry="6" fill="#FAE5C3" stroke="#D35400" stroke-width="1.5"/>
  <text x="860" y="337" text-anchor="middle" class="small" font-weight="bold">ModalityHGBlock_IR</text>
  <text x="860" y="352" text-anchor="middle" class="small">Stage 2 ×3 (96 → 256)</text>

  <rect x="790" y="390" width="140" height="45" rx="6" ry="6" fill="#D35400" stroke="#A04000" stroke-width="1.5" opacity="0.8"/>
  <text x="860" y="407" text-anchor="middle" class="white" font-weight="bold">ModalityHGBlock_IR</text>
  <text x="860" y="422" text-anchor="middle" class="white">Stage 3 ×3×3 (192 → 768)</text>

  <rect x="790" y="460" width="140" height="45" rx="6" ry="6" fill="#D35400" stroke="#A04000" stroke-width="1.5" opacity="0.8"/>
  <text x="860" y="477" text-anchor="middle" class="white" font-weight="bold">ModalityHGBlock_IR</text>
  <text x="860" y="492" text-anchor="middle" class="white">Stage 4 ×3 (384 → 1024)</text>

  <!-- IR branch arrows -->
  <line x1="860" y1="225" x2="860" y2="250" stroke="#D35400" stroke-width="1" marker-end="url(#arrowOrange)"/>
  <line x1="860" y1="295" x2="860" y2="320" stroke="#D35400" stroke-width="1" marker-end="url(#arrowOrange)"/>
  <line x1="860" y1="365" x2="860" y2="390" stroke="#D35400" stroke-width="1" marker-end="url(#arrowOrange)"/>
  <line x1="860" y1="435" x2="860" y2="460" stroke="#D35400" stroke-width="1" marker-end="url(#arrowOrange)"/>

  <!-- RDRFA Fusion modules -->
  <rect x="430" y="520" width="240" height="50" rx="8" ry="8" fill="#27AE60" stroke="#1E8449" stroke-width="2"/>
  <text x="550" y="540" text-anchor="middle" class="white" font-weight="bold">RDRFA Fusion (P5) — 1024→1024</text>
  <text x="550" y="555" text-anchor="middle" class="small" fill="#D5F5E3">Alignment → Degradation → Fusion</text>

  <rect x="430" y="590" width="240" height="50" rx="8" ry="8" fill="#27AE60" stroke="#1E8449" stroke-width="2"/>
  <text x="550" y="610" text-anchor="middle" class="white" font-weight="bold">RDRFA Fusion (P4) — 768→768</text>
  <text x="550" y="625" text-anchor="middle" class="small" fill="#D5F5E3">Alignment → Degradation → Fusion</text>

  <rect x="430" y="660" width="240" height="50" rx="8" ry="8" fill="#27AE60" stroke="#1E8449" stroke-width="2"/>
  <text x="550" y="680" text-anchor="middle" class="white" font-weight="bold">RDRFA Fusion (P3) — 256→256</text>
  <text x="550" y="695" text-anchor="middle" class="small" fill="#D5F5E3">Alignment → Degradation → Fusion</text>

  <!-- Arrows from backbone to fusion -->
  <!-- Stage 4 to P5 -->
  <line x1="310" y1="482" x2="380" y2="545" stroke="#E74C3C" stroke-width="1.5" marker-end="url(#arrowGreen)"/>
  <line x1="790" y1="482" x2="720" y2="545" stroke="#D35400" stroke-width="1.5" marker-end="url(#arrowGreen)"/>
  <!-- Stage 3 to P4 -->
  <line x1="310" y1="412" x2="380" y2="615" stroke="#E74C3C" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arrowGreen)"/>
  <line x1="790" y1="412" x2="720" y2="615" stroke="#D35400" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arrowGreen)"/>
  <!-- Stage 2 to P3 -->
  <line x1="310" y1="342" x2="380" y2="685" stroke="#E74C3C" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arrowGreen)"/>
  <line x1="790" y1="342" x2="720" y2="685" stroke="#D35400" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arrowGreen)"/>

  <!-- Head -->
  <rect x="320" y="740" width="460" height="50" rx="8" ry="8" fill="#D4E6F1" stroke="#2471A3" stroke-width="2"/>
  <text x="550" y="760" text-anchor="middle" class="label" font-weight="bold" fill="#1A5276">Hybrid Encoder (AIFI) + Bi-FPN + RTDETR Decoder</text>
  <text x="550" y="778" text-anchor="middle" class="small" fill="#1A5276">Transformer Encoder (AIFI) → Feature Pyramid → Transformer Decoder (300 queries)</text>

  <!-- Arrows from fusion to head -->
  <line x1="550" y1="710" x2="550" y2="740" stroke="#2471A3" stroke-width="2" marker-end="url(#arrowBlue)"/>
  <line x1="430" y1="670" x2="320" y2="765" stroke="#27AE60" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="670" y1="670" x2="780" y2="765" stroke="#27AE60" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- Output -->
  <rect x="400" y="810" width="300" height="45" rx="8" ry="8" fill="#1A5276" stroke="#2471A3" stroke-width="2"/>
  <text x="550" y="828" text-anchor="middle" class="white" font-weight="bold">Detection Output</text>
  <text x="550" y="844" text-anchor="middle" class="small" fill="#AED6F1">300 queries · 80 classes · NMS</text>
  <line x1="550" y1="790" x2="550" y2="810" stroke="#2471A3" stroke-width="2" marker-end="url(#arrowBlue)"/>
</svg>'''
    return svg

def create_module_svg():
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="1100" height="850" viewBox="0 0 1100 850">
  <defs>
    <marker id="arrG" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#7F8C8D"/>
    </marker>
    <marker id="arrB" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#2980B9"/>
    </marker>
    <marker id="arrR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#E74C3C"/>
    </marker>
  </defs>
  <style>
    .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #2C3E50; }
    .subtitle { font-family: Arial, sans-serif; font-size: 11px; font-weight: bold; fill: #2C3E50; }
    .text { font-family: Arial, sans-serif; font-size: 9px; fill: #2C3E50; }
    .white { font-family: Arial, sans-serif; font-size: 9px; fill: white; }
    .eq { font-family: Courier, monospace; font-size: 10px; fill: #2C3E50; }
  </style>

  <text x="550" y="25" text-anchor="middle" class="title">RDRFA Fusion Module Detail</text>
  <text x="550" y="45" text-anchor="middle" class="text" fill="#7F8C8D">Recovery-Driven Degradation-aware Fusion with Alignment</text>

  <!-- Inputs -->
  <rect x="180" y="70" width="140" height="45" rx="6" ry="6" fill="#FADBD8" stroke="#E74C3C" stroke-width="1.5"/>
  <text x="250" y="90" text-anchor="middle" class="subtitle" fill="#E74C3C">F_rgb</text>
  <text x="250" y="105" text-anchor="middle" class="text">RGB features (C)</text>

  <rect x="780" y="70" width="140" height="45" rx="6" ry="6" fill="#FDEBD0" stroke="#D35400" stroke-width="1.5"/>
  <text x="850" y="90" text-anchor="middle" class="subtitle" fill="#D35400">F_ir</text>
  <text x="850" y="105" text-anchor="middle" class="text">IR features (C)</text>

  <!-- Step 1 Box -->
  <rect x="140" y="150" width="820" height="130" rx="10" ry="10" fill="#FEF9E7" stroke="#F1C40F" stroke-width="2"/>
  <text x="550" y="175" text-anchor="middle" class="subtitle">Step 1: DeformableOffsetAlignment</text>
  
  <rect x="200" y="190" width="240" height="40" rx="6" ry="6" fill="#F7DC6F" stroke="#B7950B" stroke-width="1"/>
  <text x="320" y="210" text-anchor="middle" class="text" font-weight="bold">Conv(2C→C, 3×3) → ReLU</text>
  <text x="320" y="222" text-anchor="middle" class="text">DWConv(C→C/4, 3×3) → Tanh</text>

  <text x="480" y="210" text-anchor="middle" class="eq">offset =</text>
  <text x="550" y="210" text-anchor="middle" class="eq">Tanh(Conv(Concat(F_rgb, F_ir)))</text>

  <rect x="580" y="190" width="320" height="40" rx="6" ry="6" fill="#F7DC6F" stroke="#B7950B" stroke-width="1"/>
  <text x="740" y="207" text-anchor="middle" class="text">Grid Sample (bi-directional alignment)</text>
  <text x="740" y="222" text-anchor="middle" class="text">F_rgb' = GridSample(F_rgb, grid + offset)</text>

  <!-- Arrow -->
  <line x1="250" y1="115" x2="250" y2="150" stroke="#7F8C8D" stroke-width="1" marker-end="url(#arrG)"/>
  <line x1="850" y1="115" x2="850" y2="150" stroke="#7F8C8D" stroke-width="1" marker-end="url(#arrG)"/>

  <!-- Step 2 Box -->
  <rect x="140" y="310" width="820" height="130" rx="10" ry="10" fill="#FCF3CF" stroke="#F39C12" stroke-width="2"/>
  <text x="550" y="335" text-anchor="middle" class="subtitle">Step 2: DegradationEstimator</text>

  <rect x="200" y="350" width="700" height="40" rx="6" ry="6" fill="#F0B27A" stroke="#CA6F1E" stroke-width="1"/>
  <text x="550" y="367" text-anchor="middle" class="text">DWConv(2C→C) → DWConv(C→C/4) → Conv(C/4→5, 3×3) → Sigmoid</text>
  <text x="550" y="382" text-anchor="middle" class="text" font-weight="bold">Outputs: D_rgb, D_ir, D_blur, R_rgb, R_ir (5 interpretable maps)</text>

  <!-- Recovery -->
  <text x="250" y="475" text-anchor="middle" class="subtitle" fill="#E74C3C">F_rgb_clean = F_rgb' × R_rgb</text>
  <text x="850" y="475" text-anchor="middle" class="subtitle" fill="#D35400">F_ir_clean = F_ir' × R_ir</text>
  
  <line x1="250" y1="280" x2="250" y2="310" stroke="#7F8C8D" stroke-width="1" marker-end="url(#arrG)"/>
  <line x1="850" y1="280" x2="850" y2="310" stroke="#7F8C8D" stroke-width="1" marker-end="url(#arrG)"/>
  <line x1="250" y1="440" x2="250" y2="460" stroke="#7F8C8D" stroke-width="1" marker-end="url(#arrG)"/>
  <line x1="850" y1="440" x2="850" y2="460" stroke="#7F8C8D" stroke-width="1" marker-end="url(#arrG)"/>

  <!-- Step 3 Box -->
  <rect x="140" y="510" width="820" height="140" rx="10" ry="10" fill="#D5F5E3" stroke="#27AE60" stroke-width="2"/>
  <text x="550" y="535" text-anchor="middle" class="subtitle">Step 3: RecoveryGuidedFusion</text>

  <rect x="180" y="555" width="260" height="40" rx="6" ry="6" fill="#82E0AA" stroke="#1E8449" stroke-width="1"/>
  <text x="310" y="572" text-anchor="middle" class="text" font-weight="bold">Shared Branch</text>
  <text x="310" y="587" text-anchor="middle" class="text">SharedConv(Mix) → Fs (C/2)</text>

  <rect x="460" y="555" width="180" height="40" rx="6" ry="6" fill="#82E0AA" stroke="#1E8449" stroke-width="1"/>
  <text x="550" y="572" text-anchor="middle" class="text" font-weight="bold">Private RGB</text>
  <text x="550" y="587" text-anchor="middle" class="text">PrivConv_rgb → Fr (C/4)</text>

  <rect x="660" y="555" width="180" height="40" rx="6" ry="6" fill="#82E0AA" stroke="#1E8449" stroke-width="1"/>
  <text x="750" y="572" text-anchor="middle" class="text" font-weight="bold">Private IR</text>
  <text x="750" y="587" text-anchor="middle" class="text">PrivConv_ir → Fi (C/4)</text>

  <rect x="300" y="610" width="500" height="28" rx="6" ry="6" fill="#27AE60" stroke="#1A7A3A" stroke-width="1.5"/>
  <text x="550" y="629" text-anchor="middle" class="white" font-weight="bold">F_fusion = FusionProj( Concat[ Fs(C/2), Fr(C/4), Fi(C/4) ] ) → C channels</text>

  <!-- Output -->
  <rect x="400" y="690" width="300" height="40" rx="8" ry="8" fill="#2ECC71" stroke="#1A7A3A" stroke-width="2"/>
  <text x="550" y="710" text-anchor="middle" class="white" font-weight="bold">F_fusion (C channels) → Next Layer</text>
  <line x1="550" y1="650" x2="550" y2="690" stroke="#27AE60" stroke-width="2" marker-end="url(#arrB)"/>

  <!-- Degradation map legend -->
  <rect x="140" y="760" width="820" height="70" rx="8" ry="8" fill="#F8F9FA" stroke="#BDC3C7" stroke-width="1"/>
  <text x="550" y="780" text-anchor="middle" class="subtitle">Degradation Maps (per-pixel, value ∈ [0,1])</text>
  <rect x="200" y="793" width="120" height="22" rx="4" ry="4" fill="#FADBD8" stroke="#E74C3C"/>
  <text x="260" y="809" text-anchor="middle" class="text">D_rgb: RGB degrad</text>
  <rect x="340" y="793" width="120" height="22" rx="4" ry="4" fill="#FDEBD0" stroke="#D35400"/>
  <text x="400" y="809" text-anchor="middle" class="text">D_ir: IR degrad</text>
  <rect x="480" y="793" width="120" height="22" rx="4" ry="4" fill="#F5B7B1" stroke="#C0392B"/>
  <text x="540" y="809" text-anchor="middle" class="text">D_blur: blur level</text>
  <rect x="620" y="793" width="120" height="22" rx="4" ry="4" fill="#D5F5E3" stroke="#27AE60"/>
  <text x="680" y="809" text-anchor="middle" class="text">R_rgb: recov conf</text>
  <rect x="760" y="793" width="120" height="22" rx="4" ry="4" fill="#D5F5E3" stroke="#27AE60"/>
  <text x="820" y="809" text-anchor="middle" class="text">R_ir: recov conf</text>
</svg>'''
    return svg

# Generate SVG files
with open('figures/fig1_architecture.svg', 'w') as f:
    f.write(create_architecture_svg())
print('✅ figures/fig1_architecture.svg')

with open('figures/fig3_rdrfusion_detail.svg', 'w') as f:
    f.write(create_module_svg())
print('✅ figures/fig3_rdrfusion_detail.svg')

# Also create editable drawio file
drawio_content = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="Electron" modified="2024-01-01T00:00:00Z" agent="Mozilla/5.0" version="21.2.8">
  <diagram name="Overall Architecture" id="page1">
    <mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" page="1" pageScale="1" pageWidth="1200" pageHeight="900">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
      </root>
    </mxGraphModel>
  </diagram>
  <diagram name="RDRFusion Module" id="page2">
    <mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" page="1" pageScale="1" pageWidth="1200" pageHeight="900">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

with open('figures/RDSFA_Net_Architecture.drawio', 'w') as f:
    f.write(drawio_content)
print('✅ figures/RDSFA_Net_Architecture.drawio (template)')

print('\n🎉 Generated:')
print('   SVG files (open in browser or draw.io):')
print('   - figures/fig1_architecture.svg')
print('   - figures/fig3_rdrfusion_detail.svg')
print('   Draw.io template (open with https://app.diagrams.net):')
print('   - figures/RDSFA_Net_Architecture.drawio')
