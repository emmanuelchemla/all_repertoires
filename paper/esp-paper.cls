%% esp-paper.cls
%% Earth Species Project Paper Template
%% Compile with XeLaTeX
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{esp-paper}

% ------------------------------------------------------------------ %
%  OPTIONS
%  - external: use for joint papers with external collaborators
%              hides ESP monogram from header and "Earth Species Project"
%              from footer. Use \documentclass[external]{esp-paper}
% ------------------------------------------------------------------ %
\newif\ifespExternal
\espExternalfalse
\DeclareOption{external}{\espExternaltrue}
\ProcessOptions\relax

% ------------------------------------------------------------------ %
%  BASE CLASS
% ------------------------------------------------------------------ %
\LoadClass[a4paper,10pt,oneside]{article}

% ------------------------------------------------------------------ %
%  PACKAGES
% ------------------------------------------------------------------ %
\RequirePackage{polyglossia}
\setdefaultlanguage{english}
\RequirePackage{fontspec}
\RequirePackage{multicol}
\RequirePackage{geometry}
\RequirePackage[colorlinks=true]{hyperref}
\RequirePackage{mdframed}
\RequirePackage{fancyhdr}
\RequirePackage{authblk}
\renewcommand{\Authsep}{, }
\renewcommand{\Authands}{, }
\RequirePackage{graphicx}
\RequirePackage[table]{xcolor}
\RequirePackage{titlesec}
\RequirePackage{booktabs}
\RequirePackage{amsmath,amssymb}
\RequirePackage{xspace}
\RequirePackage[font=small,labelsep=period]{caption}
\RequirePackage[final,verbose=silent]{microtype}

% ------------------------------------------------------------------ %
%  TABLES
% ------------------------------------------------------------------ %
\renewcommand{\arraystretch}{1.4}

% ------------------------------------------------------------------ %
%  BRAND COLOURS
% ------------------------------------------------------------------ %
\definecolor{espPear}{HTML}{C3E409}
\definecolor{espPine}{HTML}{054C3B}
\definecolor{espJungle}{HTML}{129C7B}
\definecolor{espSea}{HTML}{04D78A}
\definecolor{espAmphibian}{HTML}{A7ED99}
\definecolor{espGrey}{HTML}{D5D5D5}
\definecolor{espDarkGrey}{HTML}{555555}
\definecolor{espBlack}{HTML}{00100D}

% ------------------------------------------------------------------ %
%  PAGE GEOMETRY
% ------------------------------------------------------------------ %
\geometry{
  a4paper,
  top=2.5cm, bottom=3.2cm,
  left=2.8cm, right=2.8cm,
  headsep=0.6cm,
  footskip=1.2cm,
  headheight=17pt,
}

% ------------------------------------------------------------------ %
%  FONTS
% ------------------------------------------------------------------ %
% If the proprietary TiemposText OTFs are bundled under ./fonts/ we use
% them; otherwise we fall back to a generic system serif so the paper
% still compiles for reviewers who don't have the brand fonts.
\IfFileExists{./fonts/TiemposText-Regular.otf}{
  \setmainfont{TiemposText}[
    Path           = ./fonts/,
    UprightFont    = TiemposText-Regular.otf,
    ItalicFont     = TiemposText-RegularItalic.otf,
    BoldFont       = TiemposText-Semibold.otf,
    BoldItalicFont = TiemposText-SemiboldItalic.otf,
    Ligatures      = TeX,
  ]
  \newfontfamily\headingfont{TiemposText}[
    Path           = ./fonts/,
    UprightFont    = TiemposText-Medium.otf,
    ItalicFont     = TiemposText-MediumItalic.otf,
    BoldFont       = TiemposText-Semibold.otf,
    BoldItalicFont = TiemposText-SemiboldItalic.otf,
    Ligatures      = TeX,
  ]
}{
  \ClassWarning{esp-paper}{TiemposText not found in ./fonts/; falling back to Hoefler Text}
  \setmainfont{Hoefler Text}[Ligatures = TeX]
  \newfontfamily\headingfont{Hoefler Text}[Ligatures = TeX]
}

% ------------------------------------------------------------------ %
%  TYPOGRAPHY
% ------------------------------------------------------------------ %
\setlength{\parindent}{1.5em}
\setlength{\parskip}{0pt}
\hyphenpenalty=50
\exhyphenpenalty=50
\tolerance=1414
\emergencystretch=1.5em
\widowpenalty=10000
\clubpenalty=10000
\urlstyle{same}

% ------------------------------------------------------------------ %
%  SECTION HEADINGS
% ------------------------------------------------------------------ %
\titleformat{\section}{\headingfont\large\color{espPine}\hyphenpenalty=10000\raggedright}{\thesection}{1em}{}
\titleformat{\subsection}{\headingfont\normalsize\color{espPine}\hyphenpenalty=10000\raggedright}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\headingfont\small\itshape\color{espPine}\hyphenpenalty=10000\raggedright}{\thesubsubsection}{1em}{}
\newcommand{\subsubsubsection}[1]{%
  \vspace{1em}%
  {\headingfont\small\color{espPine}\raggedright\hyphenpenalty=10000 #1}\par%
  \vspace{0.5em}%
  \@afterindentfalse\@afterheading%
}

% ------------------------------------------------------------------ %
%  ABSTRACT
% ------------------------------------------------------------------ %
\renewenvironment{abstract}{%
  \vskip 1.5em
  \begin{mdframed}[
    linecolor=espGrey,
    linewidth=1pt,
    backgroundcolor=espGrey!20!white,
    roundcorner=6pt,
    innerleftmargin=15pt,
    innerrightmargin=15pt,
    innertopmargin=12pt,
    innerbottommargin=12pt,
    skipabove=0pt,
    skipbelow=0pt,
  ]
  \centerline{\headingfont\small\color{espPine}Abstract}
  \par\vspace{0.6em}
  \small\normalfont\noindent\ignorespaces
}{%
  \par
  \end{mdframed}
  \vskip 1em
}

% ------------------------------------------------------------------ %
%  TITLE BLOCK COMMANDS
% ------------------------------------------------------------------ %
\makeatletter
\newcommand\espShortTitle{}
\newcommand\shorttitle[1]{\renewcommand\espShortTitle{#1}}
\newcommand\@venue{}
\newcommand\venue[1]{\renewcommand\@venue{#1}}
\newcommand{\espExtraLogos}{}
\newcommand{\extralogos}[1]{\renewcommand{\espExtraLogos}{#1}}

\renewcommand{\maketitle}{%
  \IfFileExists{logos/esp-logo.png}{%
    \noindent\includegraphics[height=1.4cm]{logos/esp-logo.png}%
  }{}%
  \espExtraLogos\par
  \vskip 2em
  {\noindent\headingfont\LARGE\color{espBlack}%
    \hyphenpenalty=10000\exhyphenpenalty=10000%
    \raggedright\@title\par}%
  \vskip 0.8em
  {\noindent\normalsize\@author\par}%
  \ifx\@venue\empty\else
    \vskip 0.3em{\noindent\small\itshape\color{espJungle}\@venue\par}
  \fi
  \vskip 1em
  \noindent{\color{espPine!50}\rule{\linewidth}{0.6pt}}\par
  \setcounter{footnote}{0}%
  \global\let\@thanks\@empty
}
\makeatother

% ------------------------------------------------------------------ %
%  CAPTIONS
% ------------------------------------------------------------------ %
\DeclareCaptionLabelFormat{esplabel}{{\headingfont\color{espPine}#1~#2}}
\captionsetup{labelformat=esplabel}

% ------------------------------------------------------------------ %
%  BIBLIOGRAPHY
% ------------------------------------------------------------------ %
\renewcommand{\refname}{\color{espPine}\headingfont References} 
\RequirePackage{hyperref}
\RequirePackage[style=authoryear, natbib=true, maxbibnames=99]{biblatex}
\setlength{\bibhang}{2em}
\setlength\bibitemsep{0.8em}

% Make full citation hyperlinked
\makeatletter
\def\act@on@bibmacro#1#2{%
  \expandafter#1\csname abx@macro@\detokenize{#2}\endcsname}
\def\patchbibmacro{\act@on@bibmacro\patchcmd}
\patchbibmacro{cite}{\printnames{labelname}}{\bibhyperref{\printnames{labelname}}}{}{}
\makeatother
% ------------------------------------------------------------------ %
%  HEADERS & FOOTERS
% ------------------------------------------------------------------ %
\fancypagestyle{firstpage}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0.4pt}%
  \fancyfoot[C]{\small\thepage}%
  \ifespExternal\else
    \fancyfoot[R]{\scriptsize\color{espDarkGrey}Earth Species Project}%
  \fi
}

\AtBeginDocument{%
  \pagestyle{fancy}
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0.4pt}
  \fancyhead[L]{\scriptsize\color{espPine}\MakeUppercase{\espShortTitle}}
  \ifespExternal\else
    \fancyhead[R]{\includegraphics[height=0.45cm]{logos/esp-monogram.png}}%
  \fi
  \fancyfoot[C]{\small\thepage}%
  \ifespExternal\else
    \fancyfoot[R]{\scriptsize\color{espDarkGrey}Earth Species Project}%
  \fi
  \hypersetup{
    linkcolor=espPine,
    citecolor=espJungle,
    urlcolor=espPine,
    filecolor=espPine,
  }%
}

%% End of esp-paper.cls