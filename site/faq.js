// FAQ entries. Each answer is HTML. Add new entries anywhere in the list.
const FAQ = [
  {
    q: "What operating system does ABEL run on?",
    a: "<p>Windows. ABEL runs locally on your computer.</p>",
  },
  {
    q: "Do I need a GPU?",
    a: "<p>An NVIDIA GPU is recommended. ABEL can run on CPU, but it is much slower.</p>",
  },
  {
    q: "What pose tracking files can I use?",
    a: "<p>DeepLabCut or SLEAP files, in .csv or .h5 format.</p>",
  },
  {
    q: "What video formats can I use?",
    a: "<p>.mp4 or .avi.</p>",
  },
  {
    q: "How should I name my files?",
    a: "<p>Use the format <code>AB123_conditioning</code>: experiment name, subject, then session type. Name pose and video files the same way.</p>",
  },
  {
    q: "Install fails with \"Microsoft Visual C++ 14.0 or greater is required\"",
    a: "<p>Install the <a href=\"https://visualstudio.microsoft.com/visual-cpp-build-tools/\">Microsoft C++ Build Tools</a> and select the <b>Desktop development with C++</b> workload. Then click <b>Install All Dependencies</b> in ABEL again.</p>",
  },
  {
    q: "How many clips should I label per behavior?",
    a: "<p>About 200 positive clips per behavior.</p>",
  },
  {
    q: "How do I align ABEL bouts with fiber photometry?",
    a: "<p>Export bout start and end frames from the <b>Export</b> tab. <a href=\"https://github.com/JobeRitchie/TRACY-Photometry-Processing-Suite\">TRACY</a> reads these files directly.</p>",
  },
  {
    q: "Can I use ABEL for commercial work?",
    a: "<p>ABEL is free for academic and non-profit use. For commercial use, contact the UNC Office of Technology Commercialization at 919-966-3929. See the <a href=\"https://github.com/JobeRitchie/ABEL/blob/main/LICENSE\">license</a>.</p>",
  },
  {
    q: "How do I cite ABEL?",
    a: "<p>See the <a href=\"#citation\">Citation</a> page.</p>",
  },
];
