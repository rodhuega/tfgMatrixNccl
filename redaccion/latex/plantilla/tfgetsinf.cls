\ProvidesClass{plantilla/tfgetsinf}[2015/05/04 v1 LaTeX document class]
%
% Opcions
%
\newif\iflisttables\listtablesfalse
\newif\iflistfigures\listfiguresfalse
\newif\iflistquadres\listquadresfalse
\newif\iflistalgorithms\listalgorithmsfalse

\DeclareOption{catalan}{\PassOptionsToPackage{english,spanish,main=catalan}{babel}}
\DeclareOption{spanish}{\PassOptionsToPackage{english,catalan,main=spanish}{babel}}
\DeclareOption{english}{\PassOptionsToPackage{spanish,catalan,main=english}{babel}}

\DeclareOption{listoftables}{\listtablestrue}
\DeclareOption{listoffigures}{\listfigurestrue}
\DeclareOption{listofquadres}{\listquadrestrue}
\DeclareOption{listofalgorithms}{\listalgorithmstrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions

\LoadClass{book}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Codificacio dels caracteres 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[T1]{fontenc}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sistema babel (catala/castella/angles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{babel}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tipografia base (palatino)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{mathpazo}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Geometria (a4, marges
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[a4paper,left=3cm,right=3cm,top=2.35cm,bottom=2.35cm]{geometry}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Escritura de los enlaces de internet; ejemplo: \url{www.upv.es}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{url}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{tocbibind}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Manejo de colores
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{xcolor}

%%%%%%%%%%%%%%%%%%
% Colors que fem servir en 'listings'
%%%%%%%%%%%%%%%%%%
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

%%%%%%%%%%%%%%%%
% Colors grisos (per a la portada)
%%%%%%%%%%%%%%%%
\definecolor{grisclar}{gray}{0.5}
\definecolor{grisfosc}{gray}{0.25}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Para evitar líneas huérfanas y viudas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\clubpenalty=10000
\widowpenalty=10000


%%%%%%%
% Capcalera del document
%%%%%%%

\newcommand{\tutor}[1]{\newcommand{\t@tor}{#1}}
\newcommand{\curs}[1]{\newcommand{\c@rs}{#1}}

\newcommand{\titulacioname}{Grau en Enginyeria Inform\`atica}
\newcommand{\tfgname}{Trabajo Fin de Grado}

\newcommand{\authorname}{Autor}
\newcommand{\tutorname}{Tutor}
\newcommand{\coursename}{Curs}
\newcommand{\keywordsname}{Paraules clau}

\newcommand{\algorithmname}{Algorisme}
\newcommand{\listalgorithmname}{\'Index d'algorismes}
\newcommand{\quadrename}{Quadre}
\newcommand{\listquadrename}{\'Index de quadres}

\newcommand{\appendicesname}{Ap\`endixs}

\addto\captionscatalan{%
     \renewcommand{\titulacioname}{Grau en Enginyeria Inform\`atica}
     \renewcommand{\tfgname}{Treball fi de grau}
     \renewcommand{\authorname}{Autor}
     \renewcommand{\tutorname}{Tutor}
     \renewcommand{\coursename}{Curs}
     \renewcommand{\keywordsname}{Paraules clau}
     \renewcommand{\algorithmname}{Algorisme}
     \renewcommand{\listalgorithmname}{\'Index d'algorismes}
     \renewcommand{\quadrename}{Quadre}
     \renewcommand{\listquadrename}{\'Index de quadres}
     \renewcommand{\appendicesname}{Ap\`endixs}}
     

\addto\captionsspanish{%
     \renewcommand{\titulacioname}{Grado en Ingenier\'\i a Inform\'atica}
     \renewcommand{\tfgname}{Trabajo fin de grado}
     \renewcommand{\authorname}{Autor}
     \renewcommand{\tutorname}{Tutor}
     \renewcommand{\coursename}{Curso}
     \renewcommand{\keywordsname}{Palabras clave}
     \renewcommand{\tablename}{Tabla}
     \renewcommand{\listtablename}{\'Indice de tablas}
     \renewcommand{\algorithmname}{Algoritmo}
     \renewcommand{\listalgorithmname}{\'Indice de algoritmos}
     \renewcommand{\quadrename}{Cuadro}
     \renewcommand{\listquadrename}{\'Indice de cuadros}
     \renewcommand{\appendicesname}{Ap\'endices}}
     
\addto\captionsenglish{%
     \renewcommand{\titulacioname}{Degree in Computer Engineering}
     \renewcommand{\tfgname}{Degree final work}
     \renewcommand{\authorname}{Author}
     \renewcommand{\tutorname}{Tutor}
     \renewcommand{\coursename}{Course}
     \renewcommand{\keywordsname}{Key words}
     \renewcommand{\algorithmname}{Algorithm}
     \renewcommand{\listalgorithmname}{List of algorithms}
     \renewcommand{\appendicesname}{Appendices}}
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nous tipus d'objectes flotants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Formato de los caption de tablas, figuras, etc.
% Líneas centrados, tamaño "small" para etiqueta y texto, 
% y la etiqueta en negrita
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[justification=centerlast,small,bf,labelsep=colon]{caption}

\RequirePackage{algorithmic}
\RequirePackage[within=chapter]{newfloat}

\DeclareFloatingEnvironment[fileext=loa,
     listname=\listalgorithmname,name=\algorithmname]{algorithm}
     \DeclareCaptionFormat{ruled}{\hrulefill\par#1#2#3\vspace{-.5\baselineskip}\hrulefill}
\captionsetup[algorithm]{justification=raggedright,singlelinecheck=off,position=below,format=ruled}  
   
\DeclareFloatingEnvironment[fileext=loq,
     listname=\listquadrename,name=\quadrename]{quadre}     
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Para escribir código y que quede bien
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{listings}

\lstset{ %
	language=C++,					% choose the language of the code
	basicstyle=\footnotesize,		% the size of the fonts that are used for the code
	numbers=left,                   % where to put the line-numbers
	numberstyle=\footnotesize,      % the size of the fonts that are used for the line-numbers
	stepnumber=1,                   % the step between two line-numbers. If it is 1 each line will be numbered
	numbersep=5pt,                  % how far the line-numbers are from the code
	backgroundcolor=\color{white},	% choose the background color. 
	numberstyle=\tiny\color{gray},	% choose the color of the numbers
	keywordstyle=\color{blue},		% choose the color of the keywords
	commentstyle=\color{dkgreen},	% choose the color of the comments
	stringstyle=\color{mauve}, 		% choose the color of the strings
	showspaces=false,               % show spaces adding particular underscores
	showstringspaces=false,         % underline spaces within strings
	showtabs=false,                 % show tabs within strings adding particular underscores
	frame=single,          			% adds a frame around the code
	tabsize=2,          			% sets default tabsize to 2 spaces
	captionpos=b,           		% sets the caption-position to bottom
	breaklines=true,        		% sets automatic line breaking
	breakatwhitespace=false,   		% sets if automatic breaks should only happen at whitespace
	escapeinside={\%*}{*)}          % if you want to add a comment withi your code
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Aspecto los títulos para \chapter y \section, así como 
% añadir un punto después de la numeración de las secciones
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{titlesec}

\titlelabel{\thetitle.\quad}

\titleformat{\chapter}[display]
{\normalfont\Large\filleft\sffamily\bfseries}  % FORMAT
{\titlerule[1pt]                                  
 \vspace{1pt}
 \titlerule
 \vspace{1pc}
 \LARGE\MakeUppercase{\chaptertitlename} \thechapter} % LABEL
{0pc}                                                 % SEP
{\Huge}                                               % BEFORE CODE (abans del títol)
[\vspace{5pt}{\titlerule[1pt]}]                                          % AFTER  CODE (després del títol)

\titleformat{\section}
 {\Large\normalfont\sffamily\bfseries}
 {\thesection}{0.5em}{}[\hrule height1.5pt]
%


\RequirePackage{tocloft}
%
\setlength{\cftbeforechapskip}{3pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definición de las cabeceras y pies de página
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{fancyhdr}

\pagestyle{fancy}
\setlength{\headheight}{16pt}
%\renewcommand*{\headheight}{14pt}
\renewcommand*{\headrulewidth}{1.2pt}

\renewcommand{\chaptermark}[1]{\markboth{{\sffamily\mdseries\normalsize\hspace{1em} #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{{\sffamily\mdseries\normalsize\thesection\hspace{0.5em} #1}}{}}

\lhead[\bfseries\thepage]{\sffamily\rightmark}
\rhead[\sffamily\leftmark]{\bfseries\thepage}
\cfoot{} 
\lfoot{}
\rfoot{}


\RequirePackage{graphicx}
\RequirePackage{eso-pic}


%%%%%%%%%%%%%%
%  Navegable
%%%%%%%%%%%%%

\RequirePackage[colorlinks]{hyperref}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Separación entre párrafos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setlength{\parskip}{1ex}

\newcommand{\foreign}[2]{\foreignlanguage{#1}{\textit{#2}}}
     
\newenvironment{Foreign}[1]{\begin{quote}\itshape\selectlanguage{#1}}{\end{quote}}     
      
%%%%%%%%
% Resums
%%%%%%%%

\newcommand{\keywords}[3]{%
   \addto\captionscatalan{\def\keywords{#1}}
   \addto\captionsspanish{\def\keywords{#2}}
   \addto\captionsenglish{\def\keywords{#3}}}

\newenvironment{abstract}[1][catalan]{%
        \begingroup
          \selectlanguage{#1}\addvspace{\baselineskip}\par\hfill
          {\sffamily\bfseries\Huge\abstractname}\smallskip\par}{%
          \par\nopagebreak\noindent\textbf{\keywordsname:} \keywords\vspace*{3pt}\hrule
          \iflanguage{english}{\endgroup\makeindexes}{\endgroup\par\vspace{\stretch{1}}}}


%%%%%%%%%
% Índexs
%%%%%%%%%
\newcommand{\makeindexes}{\cleardoublepage\tableofcontents\cleardoublepage
                          \iflistfigures\listoffigures\fi
                          \iflisttables\listoftables\fi
                          \iflistquadres\listofquadres\fi
                          \iflistalgorithms\listofalgorithms\fi
                          \addtocontents{toc}{\protect\noindent\protect\hrulefill\protect\par}}

\setlength{\cftaftertoctitleskip}{\baselineskip}
\setlength{\cftafterloftitleskip}{\baselineskip}
\setlength{\cftafterlottitleskip}{\baselineskip}

\renewcommand{\cfttoctitlefont}{\hfill\Huge\sffamily\bfseries}
\renewcommand{\cftlottitlefont}{\hfill\Huge\sffamily\bfseries}
\renewcommand{\cftloftitlefont}{\hfill\Huge\sffamily\bfseries}

\renewcommand{\cftaftertoctitle}{\par\noindent\hrulefill}
\renewcommand{\cftafterlottitle}{\par\noindent\hrulefill}
\renewcommand{\cftafterloftitle}{\par\noindent\hrulefill}

\newcommand{\APPENDIX}{\appendix
             \phantomsection
             \addtocontents{toc}{\protect\noindent\protect\hrulefill\protect\par}             
             \ifnum\value{appendix}=1 \addtocontents{toc}{\protect\appendixname\protect\par}
                    \else\addtocontents{toc}{\protect\appendicesname\protect\par}\fi}

\newcommand{\m@ketitle}{%
\begin{titlepage}
\AddToShipoutPicture*{\AtPageUpperLeft{%
    % Marges: dalt=esquerra=dreta=2cm 
    \raisebox{-4cm}{\hspace{2cm}\includegraphics[height=2cm]{plantilla/logo-upv}% grand� ria 116mmx37mm escalada a 62.7mmx20mm
    \hspace{5.54cm}% 21cm-(2+6.27+5.19+2)cm
                    \includegraphics[height=2cm]{plantilla/logo-etsinf}% grand� ria 109mmx42mm escalada a 51.9mmx20mm
                    }}}
\AddToShipoutPicture*{\includegraphics[width=21cm,height=12.89cm]{plantilla/baseportada}}
\centering
\vspace*{2cm}
\begin{color}{grisfosc}
\large
Escola T\`ecnica Superior d'Enginyeria Inform\`atica

Universitat Polit\`ecnica de Val\`encia
\end{color}
\vspace{\stretch{1}}

\LARGE\bfseries\@title

\large\scshape \tfgname
\par\vspace{0.2cm}\par
\normalfont\textcolor{grisclar}{\titulacioname}

\vspace{\stretch{2}}

% Autor, tutor y fecha
\begin{flushright} 
\begin{tabular}{l}
   \textit{\authorname:} \begin{tabular}[t]{l}\@author\end{tabular}\\[0.2cm]
   \textit{\tutorname:} \begin{tabular}[t]{l}\t@tor\end{tabular}
\end{tabular}
\medskip

\coursename\space\c@rs
\end{flushright}
\end{titlepage}}

\renewcommand{\cleardoublepage}{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}

\AtBeginDocument{%
    \frontmatter\m@ketitle\cleardoublepage\thispagestyle{plain}}

\newcounter{appendix}
\AtEndDocument{%
    \ifnum\value{chapter}=1 
           \immediate\write\@auxout{\unexpanded{\setcounter{appendix}{1}}}
    \else 
           \immediate\write\@auxout{\unexpanded{\setcounter{appendix}{2}}} \fi}