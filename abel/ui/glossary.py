"""Glossary behind the Info > Glossary search.

One entry per button, setting, tab or term a user meets in ABEL. Each entry says
what the thing does in plain words and where to find it. Keep entries short,
avoid jargon unless the term itself is jargon, and never use em dashes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    where: str
    text: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    see: tuple[str, ...] = field(default_factory=tuple)


def _e(
    term: str,
    where: str,
    text: str,
    aliases: tuple[str, ...] = (),
    see: tuple[str, ...] = (),
) -> GlossaryEntry:
    return GlossaryEntry(term, where, " ".join(text.split()), aliases, see)


ENTRIES: tuple[GlossaryEntry, ...] = (
    # ── Main tabs ────────────────────────────────────────────────────────
    _e("Home", "Main tab",
       "The start page. Create or open a project, see its stats, and free up "
       "disk space.",
       see=("Project",)),
    _e("Data Import", "Main tab",
       "Add videos and pose files, pair each video with its pose file, and set "
       "each session's subject, session type and px/mm.",
       see=("Session", "Auto Match", "px/mm")),
    _e("Behaviors", "Main tab",
       "Define the behaviors you want to score: name, short name, color, hotkey "
       "and a written definition reviewers can work from.",
       aliases=("Behaviors tab",), see=("Behavior", "No Behavior")),
    _e("ROI", "Main tab",
       "Region of interest. A zone you draw on the video frame, such as an open "
       "arm or a chamber. ABEL builds zone features from it and reports time in "
       "zone.",
       aliases=("Region of interest", "ROI Definition"),
       see=("Target zone", "Subject Crop", "Advanced ROI features")),
    _e("Features", "Main tab",
       "Choose which pose, context and video measurements to compute, set the "
       "window and stride, then extract them.",
       aliases=("Features tab",), see=("Feature", "Window duration", "Stride")),
    _e("Active Learning", "Main tab",
       "The labeling loop. The model picks the clips you should label next, "
       "mostly the ones it is least sure about, so a good model takes far fewer "
       "labels than labeling at random. Add seeds, train, extract clips, label "
       "them and retrain. Holds the Seeds, Learning, Clips and Review subtabs.",
       aliases=("AL",), see=("Seeds", "Learning", "Clips", "Review (Active Learning)")),
    _e("Temporal", "Main tab",
       "Turns window scores into frame-by-frame bouts. Holds the Refinement and "
       "Review subtabs.",
       see=("Temporal refinement", "Bout")),
    _e("Analytics", "Main tab",
       "Plots and statistics on the detected bouts: counts, durations, latency, "
       "distance, heatmaps, transitions, motifs, HMM states and social measures.",
       aliases=("Behavior Analytics",), see=("Factor", "Statistics")),
    _e("Validation", "Main tab",
       "Measures how good the models are and how consistent the labels are: "
       "model overview, blind quizzes, behavior grids, feature audit and review "
       "effort.",
       see=("Leave-one-mouse-out CV", "Validation Quiz")),
    _e("Export", "Main tab",
       "Writes spreadsheets, Word tables, boutframes workbooks, TRACY position "
       "files and labeled videos for use outside ABEL.",
       see=("Boutframes workbook", "TRACY")),
    _e("Direct Use", "Main tab",
       "Apply a trained workflow to new videos without labeling again. Holds the "
       "Run, Apply Models and Transfer Feedback subtabs.",
       see=("Workflow snapshot", "Transfer Feedback")),
    _e("Model Refinement", "Main tab",
       "Import labeled examples and models from another ABEL project and map their "
       "behaviors onto this one, so you can train on both.",
       see=("Import Project as Baseline", "Map Behaviors")),
    _e("Settings", "Main tab",
       "App preferences, Python package installs and the log viewer. Holds the "
       "Preferences, Dependencies, Logs and Help subtabs.",
       see=("Preferences", "Dependencies")),
    _e("Info", "Main tab",
       "This tab. Holds the glossary, version and updates, the methods reference "
       "and the version history.",
       see=("Glossary", "Methods")),

    # ── Subtabs ──────────────────────────────────────────────────────────
    _e("Seeds", "Active Learning > Seeds",
       "Mark exact start and end frames of a behavior in a full video. Seeds give "
       "the first model something to learn from before any clips are reviewed.",
       aliases=("Seed", "Seed example"), see=("Mark Start", "Add Seed")),
    _e("Learning", "Active Learning > Learning",
       "Train models, score every window, and pick the next clips to review. Also "
       "holds the model charts and analysis tools.",
       see=("Run Pipeline", "Retrain", "Selection mode")),
    _e("Clips", "Active Learning > Clips",
       "Cut short video clips for the windows the model picked, so you can watch "
       "and label them in Review.",
       aliases=("Clip Extraction",), see=("Extract Clips", "Candidate")),
    _e("Review (Active Learning)", "Active Learning > Review",
       "Watch each clip and label it. Your labels become training data the next "
       "time you retrain.",
       aliases=("Review", "Clip Review", "Review tab"),
       see=("Accept", "Reject", "Behavior Soundboard")),
    _e("Refinement", "Temporal > Refinement",
       "Runs every model over every session at a fine time step to build "
       "probability traces, then turns them into bouts.",
       aliases=("Temporal Refinement tab",), see=("Temporal refinement", "Inference step")),
    _e("Review (Temporal)", "Temporal > Review",
       "Look at each behavior's probability trace against the video, set the "
       "threshold, minimum bout and merge gap, and flag misses and false alarms.",
       aliases=("Temporal Review",), see=("Per-Behavior Thresholds", "Flag FP", "Flag FN")),
    _e("Summary & Statistics", "Analytics > Summary & Statistics",
       "A table of every session with its group factors and per-behavior results. "
       "Assign groups here and run statistics.",
       aliases=("Summary",), see=("Factor", "Statistics")),
    _e("Graphs", "Analytics > Graphs",
       "Bar, line and time-course charts of bout count, duration, mean bout "
       "duration, latency and distance, split by your group factors.",
       see=("Bout Count", "Latency", "Time bin")),
    _e("Spatial Heatmap", "Analytics > Spatial Heatmap",
       "Shows where in the arena each behavior happened, drawn over a picture of "
       "the empty arena.",
       see=("Bandwidth", "Clean plate")),
    _e("Density Analysis", "Analytics > Density Analysis",
       "Smoothed maps of where behaviors happen, per group, plus a map of the "
       "difference between two groups. Holds the Density Maps and Group "
       "Comparison subtabs.",
       aliases=("Density Maps", "Group Comparison"),
       see=("Bandwidth", "Difference map")),
    _e("Behavior Relationships", "Analytics > Behavior Relationships",
       "How behaviors follow each other in time. Holds the Transition Matrix, "
       "motif and HMM Analysis views.",
       see=("Transition matrix", "Motif", "HMM")),
    _e("Session Sections", "Analytics > Session Sections",
       "Split each session into named time blocks, such as tone and ITI periods, "
       "and compare behavior between blocks.",
       aliases=("Section",), see=("ITI", "Section preset")),
    _e("Velocity", "Analytics > Velocity",
       "Speed of the animal during the bouts of one behavior, including the speed "
       "profile before, during and after each bout.",
       see=("Velocity profile", "Outlier Detection")),
    _e("Social Interaction", "Analytics > Social Interaction",
       "Measures between two animals in the same session: distance, approach, "
       "displacement and dominance.",
       see=("Dyad", "Displacement", "Dominance HMM")),
    _e("Overview", "Validation > Overview",
       "A table of every trained model with its held-out scores and label counts, "
       "plus the leave-one-mouse-out and session quality tools.",
       aliases=("Model Overview",), see=("PR-AUC", "F1")),
    _e("Validation Quiz", "Validation > Validation Quiz",
       "A blind test. ABEL builds a random set of clips, a reviewer labels them "
       "without seeing the model's answer, and the answers are compared with the "
       "model and with other reviewers.",
       aliases=("Quiz",), see=("Generate Test", "Cohen's kappa")),
    _e("Results & Suggestions", "Validation > Results & Suggestions",
       "Scores from finished quizzes, agreement between reviewers, and suggested "
       "fixes such as adding inhibition between two behaviors that get confused.",
       see=("Cohen's kappa", "Inhibition weight")),
    _e("Behavior Grid", "Validation > Behavior Grid",
       "A 5 x 5 video montage of 25 detected bouts of one behavior with pose dots "
       "on top. A quick way to see what the model calls that behavior.",
       aliases=("Generate Grid",),
       see=("Strongest bouts", "Probability bands")),
    _e("Feature Audit", "Validation > Feature Audit",
       "Checks which body parts were tracked and which features are dead or weak, "
       "and shows which features the trained models rely on most.",
       see=("Dead feature", "Feature importance", "Apply Recommended Exclusions")),
    _e("Review Effort", "Validation > Review Effort",
       "Measures how much human time the clip review took, from the timestamps of "
       "each decision, with breaks left out.",
       see=("Measure Review Effort",)),
    _e("Run", "Direct Use > Run",
       "Step-by-step Direct Use: pick a source project, add new videos and pose "
       "files, calibrate, map keypoints, draw ROIs and run.",
       see=("Direct Use", "Run Direct Use Pipeline")),
    _e("Apply Models", "Direct Use > Apply Models",
       "Run trained models from another project on this project's data, after "
       "checking that the features match.",
       see=("Import Selected Models",)),
    _e("Transfer Feedback", "Direct Use > Transfer Feedback",
       "Checks how well a model carried over to new data. Ranks subjects worst "
       "first and flags near-zero detections, outliers, confidence stuck high or "
       "lost, and profiles unlike the source project.",
       see=("Analyze Transfer",)),
    _e("Preferences", "Settings > Preferences",
       "App-wide options such as autosave interval, how many recent projects to "
       "list, and whether to check for updates on startup.",
       see=("Autosave seconds",)),
    _e("Dependencies", "Settings > Dependencies",
       "Shows which Python packages are installed and installs missing ones, "
       "including the GPU build of PyTorch.",
       see=("Install GPU PyTorch", "Run Smoke Test")),
    _e("Logs", "Settings > Logs",
       "The app's running log. Useful when something fails and you need the "
       "error text.",
       aliases=("Refresh Logs", "Log"),),
    _e("Help", "Settings > Help",
       "A short walk through the tabs in the order you use them.",),
    _e("Glossary", "Info > Glossary",
       "This search. Type any word you see in ABEL to get a short explanation.",),
    _e("About & Updates", "Info > About & Updates",
       "Shows the installed version and lets you check for and install updates "
       "from GitHub.",
       see=("Check for Updates", "Install Update")),
    _e("Methods", "Info > Methods",
       "The references, formulas and write-up helper for describing ABEL in a "
       "paper.",
       see=("References", "Formulas", "Write-up Helper")),
    _e("References", "Info > Methods > References",
       "The published sources behind each method ABEL uses, with links.",),
    _e("Formulas", "Info > Methods > Formulas",
       "The equations ABEL computes, each tied to the function that computes it.",),
    _e("Write-up Helper", "Info > Methods > Write-up Helper",
       "Drafts a methods section from this project's own settings and results. "
       "It leaves [FILL IN] where ABEL cannot know the answer. Check and edit it "
       "before use.",
       aliases=("Generate Draft", "Keep the draft warning at the top of the text"),),
    _e("Version History", "Info > Version History",
       "What changed in each release.",
       aliases=("Changelog",)),

    # ── Project and data ─────────────────────────────────────────────────
    _e("Project", "Home",
       "A folder that holds one experiment: its settings, sessions, behaviors, "
       "labels, models and outputs. Derived files go in derived/ and exports in "
       "exports/.",
       aliases=("Project folder", "Open Project Folder"),
       see=("Create New Project", "derived folder")),
    _e("Create New Project", "Home",
       "Makes a new project folder. You set the assay, species, number of "
       "animals, frame rate, default clip length and crop margin.",
       aliases=("New project",), see=("Assay", "Number of animals")),
    _e("Open Project", "Home",
       "Open an existing ABEL project folder. Double-click a recent project to "
       "open it quickly.",
       aliases=("Open Another Project", "Recent projects"),),
    _e("Assay", "Create New Project",
       "The behavioral test the project records, such as open field, elevated "
       "plus maze or fear conditioning. Used to suggest behaviors and presets.",),
    _e("Session", "Data Import",
       "One recording: a video plus its pose file. A subject can have many "
       "sessions, for example one per test day.",
       aliases=("Session ID", "Sessions"), see=("Subject", "Session type")),
    _e("Subject", "Data Import",
       "The animal a session records. Used for grouping, for spreading clips "
       "across animals, and for leave-one-mouse-out testing.",
       aliases=("Subject ID", "Mouse", "Animal"), see=("Session",)),
    _e("Session type", "Data Import",
       "The kind of session, such as conditioning, extinction or recall. Read "
       "from the file name and used to group sessions.",
       aliases=("Day label",), see=("Auto-Group by Session Type",)),
    _e("Pose file", "Data Import",
       "The tracking output from DeepLabCut or SLEAP: the x, y position and "
       "confidence of each body part in every frame.",
       aliases=("Pose", "Tracking file", "DLC file"),
       see=("Keypoint", "DeepLabCut", "SLEAP")),
    _e("Keypoint", "Data Import",
       "One tracked body part, such as the nose or tail base.",
       aliases=("Body part", "Detected Body Parts", "Show Keypoints"),
       see=("Keypoint Mapping", "Rename Body Parts")),
    _e("DeepLabCut", "Data Import",
       "Outside software that tracks body parts in video. ABEL reads its .csv and "
       ".h5 output.",
       aliases=("DLC",), see=("Pose file", "SLEAP")),
    _e("SLEAP", "Data Import",
       "Outside software that tracks body parts in video, often used for several "
       "animals at once. ABEL reads its output.",
       see=("Pose file", "DeepLabCut")),
    _e("Add Video(s)", "Data Import",
       "Add video files (.mp4 or .avi) to the project.",
       aliases=("Import Videos",),),
    _e("Import Pose (DLC / SLEAP)", "Data Import",
       "Add pose tracking files to the project.",
       aliases=("Add DLC Pose File(s)",), see=("Pose file",)),
    _e("Auto Match", "Data Import",
       "Pairs each video with its pose file by name and reads the subject and "
       "session type from the file name.",
       see=("Filename pattern",)),
    _e("Filename pattern", "Data Import",
       "The rule that pulls the subject and session type out of a file name. Use "
       "Build Pattern from Filename to highlight them in an example name instead "
       "of writing the rule by hand.",
       aliases=("Build Pattern from Filename", "Test Pattern", "Parsing Settings",
                "Apply Parsing Settings", "Use as Subject", "Use as Session",
                "No Session in Name"),),
    _e("Save Import Manifest", "Data Import",
       "Saves the list of sessions, their files, subjects and settings to the "
       "project.",
       aliases=("Import manifest", "Manifest"),),
    _e("px/mm", "Data Import",
       "Pixels per millimeter. Converts pixel distances into real units so speed "
       "and distance come out in cm. Each camera setup needs its own value.",
       aliases=("Pixels per mm", "Calibrate px/mm", "Calibration (px/mm)",
                "Apply px/mm", "Apply px/mm to All Sessions",
                "Apply px/mm to Selected", "Calibrate (2-point)"),
       see=("Calibrate (2-point)",)),
    _e("Calibrate (2-point)", "Data Import, Direct Use",
       "Click two points a known distance apart on a video frame and enter that "
       "distance. ABEL computes px/mm from it.",
       aliases=("Two-point calibration",), see=("px/mm",)),
    _e("Copy Files to Project", "Data Import",
       "Copies every video and pose file into the project's raw/ folder, so the "
       "project no longer depends on an outside drive.",),
    _e("Set Video Path", "Data Import",
       "Points the project at a new folder for its videos, matching files by "
       "name. Use it when the videos moved.",
       aliases=("Relocate videos",),),
    _e("Set DLC Path", "Data Import",
       "Points the project at a new folder for its pose files, matching by name.",
       aliases=("Relocate pose files",),),
    _e("Replace Pose Files", "Data Import",
       "Swaps in pose files from a new DeepLabCut or SLEAP run. Labels and animal "
       "ids are kept, and the sessions are extracted again.",),
    _e("Remove Selected Session(s)", "Data Import",
       "Removes sessions from the project, along with the features and caches "
       "built from them.",),
    _e("Number of animals", "Data Import, Create New Project",
       "How many animals are tracked per session. More than one turns on "
       "per-animal features and social features.",
       aliases=("Multi-animal", "Single-animal", "Set number of animals"),
       see=("Social features",)),
    _e("Map Animal Identities", "Data Import",
       "For multi-animal files, gives each tracked animal (Mouse1, Mouse2) a real "
       "name, such as black or white. That name is its animal id everywhere.",
       aliases=("Animal identity", "Identities", "Animal id"),
       see=("Identity swap",)),
    _e("Identity swap", "Data Import > Map Animal Identities",
       "When the tracker mixes up two animals partway through a video, so each "
       "track follows the wrong animal from then on.",
       aliases=("Swap", "Suspected swaps", "Next Swap", "Prev Swap"),
       see=("Mark Swap at This Frame", "Scan Swaps")),
    _e("Mark Swap at This Frame", "Data Import > Map Animal Identities",
       "Exchanges the two animals' tracks from this frame to the end of the "
       "session. Press it again on the same frame to undo.",),
    _e("Auto-Detect Swaps from Appearance", "Data Import > Map Animal Identities",
       "When the animals differ in coat color, reads the pixels at each track to "
       "tell them apart and proposes swap fixes. It says so when the animals look "
       "too alike to tell.",),
    _e("Scan Swaps", "Data Import",
       "Checks every multi-animal session against the video and lists sessions "
       "whose saved swap fixes disagree with what the coat colors show.",
       aliases=("Scan Swaps (appearance)",),),
    _e("Keypoint Mapping", "Data Import, Direct Use",
       "Matches body part names that differ between files or projects, such as "
       "back_mid and center_body, so the model finds the parts it expects.",
       aliases=("Auto-map", "Keypoint map"),),
    _e("Rename Body Parts", "Data Import",
       "Gives body parts new names that all later steps use.",),

    # ── Behaviors ────────────────────────────────────────────────────────
    _e("Behavior", "Behaviors",
       "Something you want ABEL to detect, such as rearing or freezing. Each "
       "behavior gets its own model.",
       see=("No Behavior", "Behavior preset")),
    _e("No Behavior", "Behaviors, Review",
       "The catch-all label for clips that show none of your behaviors or are "
       "unclear. Every project needs it. A No Behavior label counts as a negative "
       "for every behavior's model.",
       aliases=("no_behavior", "Universal negative"),
       see=("Negative",)),
    _e("Hotkey", "Behaviors",
       "The key that applies a behavior's label in Review. Each behavior needs a "
       "different key.",
       aliases=("Shortcut", "Short name"),),
    _e("Behavior preset", "Behaviors",
       "A starting set of behavior definitions for a common assay. Applying one "
       "only adds behaviors you do not already have.",
       aliases=("Apply Preset", "Save as Preset", "Preset (behaviors)"),),
    _e("Co-occurring labels", "Behaviors",
       "Lets one clip carry more than one behavior label, such as grooming and "
       "rearing at once. Each label becomes its own training row.",
       aliases=("Allow co-occurring behavior labels", "Multi-label"),),
    _e("Social behavior", "Behaviors",
       "A behavior between two animals, such as chasing. It is trained with the "
       "social features and needs a multi-animal project.",
       aliases=("Social / interaction behavior", "Interaction behavior"),
       see=("Directionality",)),
    _e("Directionality", "Behaviors",
       "For a social behavior. Directed labels only the animal doing it (the "
       "actor). Mutual labels both animals. None is a solo behavior.",
       aliases=("Directed", "Mutual", "Actor", "Recipient"),),
    _e("Repair No Behavior", "Behaviors",
       "Shown when the built-in No Behavior was renamed into a real behavior. "
       "Gives that behavior its own id and restores the No Behavior label. Files "
       "are backed up first.",
       aliases=("Repair “No Behavior”",),),

    # ── ROI ──────────────────────────────────────────────────────────────
    _e("Target zone", "ROI",
       "A zone you draw for the assay, such as an open arm. ABEL measures where "
       "the animal is relative to it.",
       aliases=("Target Zone ROIs", "Zone", "Clear All Target Zones"),
       see=("ROI",)),
    _e("Subject Crop", "ROI",
       "A rectangle drawn around the area the animal can reach, saved with each "
       "subject's ROIs.",
       aliases=("Subject Crop ROI", "Draw Subject Crop"),),
    _e("Project default", "ROI",
       "One set of ROIs used for every subject. Pick Subject override instead when "
       "the maze is not in the same spot in every video.",
       aliases=("Subject override", "Per-subject override", "Apply to"),),
    _e("Copy Current ROI to All Subjects", "ROI",
       "Copies the zones and crop on screen to every subject. Use it when every "
       "video has the same camera position.",
       aliases=("Copy Current to All Subjects",),),
    _e("Smooth (ROI)", "ROI",
       "Rounds off the corners of a hand-drawn zone.",
       aliases=("Smooth",),),
    _e("Angularize", "ROI",
       "Simplifies a hand-drawn zone to fewer, straighter corners.",),
    _e("Compact Polygon ROIs", "ROI",
       "Removes extra points from old hand-drawn zones. The shapes stay the same "
       "but the file gets much smaller and switching subjects gets faster.",),
    _e("ROI Day Exclusions", "ROI",
       "Day labels, such as Acclimation, whose sessions get no ROI features. Use "
       "it for days the zones do not apply.",),
    _e("Clear All ROI Data", "ROI",
       "Deletes every project and per-subject ROI so you can start over.",),
    _e("Load Frame", "ROI",
       "Shows a frame from the selected session's video to draw on.",
       aliases=("Auto-Load Subject Video",),),

    # ── Features ─────────────────────────────────────────────────────────
    _e("Feature", "Features",
       "One number that describes a window of time, such as average nose speed or "
       "distance to a zone. The model learns behaviors from these numbers.",
       aliases=("Features", "Feature column"),
       see=("Window", "Pose features", "Context features")),
    _e("Window", "Features",
       "A short stretch of video, such as 0.5 s, that ABEL scores as one unit. "
       "Features are computed per window and each window gets a probability for "
       "each behavior.",
       aliases=("Segment", "Feature window", "Segment window"),
       see=("Window duration", "Stride")),
    _e("Window duration", "Features",
       "How long each window is. Short windows suit brief behaviors. A rough rule "
       "is twice the length of the shortest behavior.",
       aliases=("Window size", "Clip duration"), see=("Stride",)),
    _e("Stride", "Features",
       "Time from the start of one window to the start of the next. A stride "
       "shorter than the window makes windows overlap. Smaller strides give finer "
       "timing and more data.",
       aliases=("Stride (overlap)", "Overlap"), see=("Window duration",)),
    _e("Estimate Window Count", "Features",
       "Shows how many windows the current window and stride will make.",),
    _e("Source FPS", "Features",
       "Frames per second of the videos. Used to turn seconds into frames.",
       aliases=("FPS", "Frame rate", "Default FPS", "FPS override"),),
    _e("Min likelihood", "Features",
       "The tracker's confidence cut-off. Body part positions below it are "
       "treated as missing.",
       aliases=("Likelihood", "Confidence (pose)"),),
    _e("Interpolate short pose dropouts", "Features",
       "Fills short gaps in tracking by drawing a straight line between the last "
       "and next good positions.",
       aliases=("Interpolation", "Interpolate"),),
    _e("Smoothing window", "Features",
       "Averages each body part position over this many frames to remove "
       "jitter.",
       aliases=("Pose smoothing",),),
    _e("Pose features", "Features",
       "Features computed from body part positions only: speeds, angles, "
       "distances between parts.",
       aliases=("Kinematics", "Kinematic features"),),
    _e("Context features", "Features",
       "Features about the animal's surroundings: position relative to ROIs, "
       "local motion and optical flow.",
       see=("Local motion", "Optical flow")),
    _e("Include video-derived features", "Features",
       "Computes features from the video pixels as well as from pose, such as "
       "optical flow and local motion. Slower. Pose-only is often enough for "
       "freezing or rearing.",
       aliases=("Video features", "use_video_features"),),
    _e("Optical flow", "Features",
       "How the pixels move from one frame to the next. Picks up small motions, "
       "such as paw or whisker movement, that tracking can miss.",
       aliases=("Flow", "Flow temporal stride"),),
    _e("Local motion", "Features",
       "How much the pixels change in a small box that follows the animal. Picks "
       "up body movement and disturbed bedding.",
       aliases=("Local Motion Settings",), see=("Local radius", "BG sensitivity threshold")),
    _e("Local radius", "Features",
       "Half the width, in pixels, of the box that follows the animal for local "
       "motion features. Larger takes in more of the floor around it.",
       aliases=("Local radius (px)", "Radius (px)"),),
    _e("BG sensitivity threshold", "Features",
       "How different a pixel must be from the recent background to count as "
       "moving. Lower is more sensitive. In Preview, a still animal should show "
       "little green.",
       aliases=("Background subtraction", "BG subtract", "MOG2"),),
    _e("Preview (Features)", "Features",
       "Side-by-side video of raw tracking and tracking with your smoothing, gap "
       "filling and local motion settings applied.",
       aliases=("Generate Preview", "Preview Video Settings"),),
    _e("R3D", "Features, Learning settings",
       "A video network (R3D-18) trained on general video. ABEL runs it on a crop "
       "around the animal and adds 512 numbers per window that describe what the "
       "animal looks like. Helps for digging, grooming and bedding. Slow without "
       "a GPU.",
       aliases=("R3D-18", "R3D appearance features", "Appearance embedding",
                "R3D appearance embeddings"),
       see=("Embedding",)),
    _e("Per-keypoint kinematics", "Features",
       "Velocity, speed, acceleration and jerk (change in acceleration) for each "
       "body part.",
       aliases=("Velocity", "Acceleration", "Jerk"),),
    _e("Global movement", "Features",
       "Whole-body speed from the centroid, plus forepaw and nose speed.",
       aliases=("Centroid", "Centroid velocity"),),
    _e("Oscillation / rhythm", "Features",
       "Detects repeated motion, such as paw strokes in grooming, from how a "
       "signal repeats over time.",
       aliases=("Autocorrelation", "FFT", "Rhythm"),),
    _e("Spatial / orientation", "Features",
       "Head pitch and the direction the body points.",),
    _e("Egocentric kinematics", "Features",
       "Speed forward and sideways relative to the animal's own body, so features "
       "do not depend on camera angle or which way the animal faces.",
       aliases=("Egocentric",),),
    _e("Body-length normalization", "Features",
       "Divides distances by the animal's nose-to-tail length, so features do not "
       "depend on animal size or camera zoom.",
       aliases=("body_length_px",),),
    _e("Relative geometry", "Features",
       "Distances between every pair of body parts. Describes posture. Adds many "
       "columns when many body parts are tracked.",),
    _e("Head direction", "Features",
       "Which way the head points and how fast it turns. Useful for orienting and "
       "investigation.",),
    _e("Joint angles", "Features",
       "Angles at body joints, such as spine bend or limb angles. Helps with "
       "rearing, grooming and walking.",),
    _e("Spine curvature", "Features",
       "How bent the spine is along the midline body parts. Needs at least three "
       "midline points.",),
    _e("Clip-wise deltas", "Features",
       "How angles and distances change from the start to the end of a window, "
       "not just their average. Catches an animal rising or two parts closing in.",
       aliases=("Delta", "Trend"),),
    _e("Social features", "Features",
       "For each animal, distance, facing angle, approach speed and body overlap "
       "with the other animals. Needs a multi-animal project.",
       aliases=("Social / Interaction (multi-animal)", "social_*"),),
    _e("Advanced ROI features", "Features",
       "Adds whether the animal is inside each zone, its distance to the zone "
       "edge and corners, and where it is along the zone. Without these a zone is "
       "just its center point.",),
    _e("Extract Pose Features", "Features",
       "Computes all selected features for the chosen sessions. Can take minutes "
       "to hours.",
       aliases=("Extract features", "Feature extraction", "Run (Features)"),),
    _e("Suggest Settings", "Features",
       "Recommends window, stride and smoothing from your behavior definitions "
       "and sessions.",
       aliases=("Suggest Settings From Data",),),
    _e("Clear Cached Features", "Features",
       "Deletes saved features so the next extraction rebuilds them from scratch. "
       "Your videos, pose files and settings are not touched.",
       aliases=("Clear cache first",), see=("Cache",)),
    _e("Select Needing Extraction", "Features",
       "Ticks only the sessions whose saved features are missing or out of date.",),
    _e("Re-cut review clips for re-processed sessions", "Features",
       "After features are rebuilt for a session, cuts its review clips again so "
       "they match.",),
    _e("Cache", "Throughout",
       "Results saved to disk so they are not computed twice. Clear a cache when "
       "you change settings and want results rebuilt.",
       aliases=("Cached", "Reuse cached features", "Stale cache"),),
    _e("Configure Features", "Learning",
       "Choose which feature columns the model may train on.",
       aliases=("Feature Selection",),),
    _e("Dead feature", "Validation > Feature Audit",
       "A feature with no variation, or missing in over 99% of rows, often "
       "because its body part is not tracked. It cannot help the model. A weak "
       "feature is nonzero in under 5% of rows.",
       aliases=("Weak feature", "Feature Health Summary"),),
    _e("Apply Recommended Exclusions", "Validation > Feature Audit",
       "Leaves the dead features out of future training runs.",
       aliases=("Run Feature Audit",),),
    _e("Feature importance", "Validation > Feature Audit",
       "How much each feature helped a trained model make its decisions.",
       aliases=("Refresh Importance Scores",),),

    # ── Machine learning terms ──────────────────────────────────────────
    _e("Machine learning", "Concept",
       "Teaching a computer to recognize a pattern from labeled examples instead "
       "of writing the rule by hand.",
       aliases=("ML",), see=("Model", "Training")),
    _e("Model", "Concept",
       "The trained rule that takes a window's features and returns a probability "
       "that a behavior is happening. ABEL trains one model per behavior.",
       aliases=("Classifier", "Behavior model", "Saved models"),
       see=("XGBoost", "Training")),
    _e("XGBoost", "Learning",
       "The kind of model ABEL uses: many small decision trees, each one "
       "correcting the errors of the ones before.",
       aliases=("Gradient boosting", "Decision trees", "Trees"),),
    _e("One-vs-rest", "Learning",
       "Each behavior's model answers one yes-or-no question: is this behavior "
       "happening or not. The answers are compared afterwards.",
       aliases=("Binary classifier",),),
    _e("Training", "Learning",
       "Fitting a model to your labeled windows so it learns which feature values "
       "go with each behavior.",
       aliases=("Train", "Training set", "Training data"),),
    _e("Inference", "Learning, Temporal",
       "Using a trained model to score windows it has not seen. Produces a "
       "probability per window.",
       aliases=("Prediction", "Scoring", "Run Inference"),),
    _e("Probability", "Throughout",
       "The model's confidence, from 0 to 1, that the behavior is happening in a "
       "window or frame.",
       aliases=("Score", "Confidence", "prediction_prob"),
       see=("Threshold", "Calibration")),
    _e("Label", "Review, Seeds",
       "Your answer for a clip or window: which behavior it shows, or that it "
       "shows none. Labels are what the model learns from.",
       aliases=("Labels", "Reviewer labels", "Annotation"),),
    _e("Positive", "Concept",
       "A labeled example where the behavior is happening.",
       aliases=("Positive example", "Positives"),),
    _e("Negative", "Concept",
       "A labeled example where the behavior is not happening. A rejected clip is "
       "a negative for the rejected behavior only. A No Behavior clip is a "
       "negative for all of them.",
       aliases=("Negative example", "Negatives"),),
    _e("Hard negative", "Concept",
       "A clip labeled as not the behavior that the model still scores high. "
       "These are the model's most confident mistakes and are the most useful to "
       "review.",
       aliases=("Hard-negative mining", "hard_negative"),),
    _e("Held-out data", "Learning",
       "Labeled windows kept out of training and used only to test the model. "
       "Tests on held-out data show how well the model handles new data.",
       aliases=("Holdout", "Validation holdout", "Validation set", "Test set"),
       see=("Split strategy",)),
    _e("Split strategy", "Learning settings",
       "How labeled data is divided into training and held-out sets. Group by "
       "subject keeps all of an animal's windows on one side, which avoids "
       "leakage when there are 3 or more subjects.",
       aliases=("Group by subject", "Group by session"),
       see=("Data leakage",)),
    _e("Data leakage", "Concept",
       "When test data is too similar to training data, such as overlapping "
       "windows from the same animal on both sides. Scores then look better than "
       "the model really is.",
       aliases=("Leakage",),),
    _e("Overfitting", "Concept",
       "When a model learns its training examples too closely and does worse on "
       "new data.",),
    _e("Generalization", "Concept",
       "How well a model works on animals or sessions it was not trained on.",),
    _e("Calibration", "Learning",
       "Adjusting a model's scores so a probability of 0.7 really means right "
       "about 70% of the time. ABEL uses Platt scaling.",
       aliases=("Platt scaling", "Calibrated"),),
    _e("Temperature scale", "Learning settings, Temporal > Refinement",
       "Softens or sharpens probabilities after scoring. Above 1 pulls them "
       "toward 0.5, which helps a model that is too sure of itself. 1 changes "
       "nothing.",
       aliases=("Temperature", "Probability temperature", "Temperature scaling"),),
    _e("Augmentation", "Learning",
       "Adds slightly altered copies of positive examples during training (small "
       "random noise, some features blanked) so the model does not lean on exact "
       "values.",
       aliases=("Positive augmentation",),),
    _e("Class imbalance", "Concept",
       "When one label is far more common than another, such as many more No "
       "Behavior windows than grooming windows.",
       see=("No-behavior sample weight", "PR-AUC")),
    _e("No-behavior sample weight", "Learning settings",
       "How much No Behavior examples count in training. Raise it to cut false "
       "positives. 0 sets it from the label counts.",),
    _e("Training cap per class", "Learning settings",
       "Most labeled examples per label used in training. Keeps very large "
       "projects fast. 0 uses all.",),
    _e("Threshold", "Temporal > Review",
       "The probability a frame must reach to count as the behavior. Lower finds "
       "more bouts but also more false ones.",
       aliases=("Onset threshold", "Positive threshold", "Cut-off"),
       see=("Per-Behavior Thresholds",)),
    _e("True positive", "Concept",
       "The model said yes and the label says yes.",
       aliases=("TP",),),
    _e("False positive", "Concept",
       "The model said yes but the label says no. A false alarm.",
       aliases=("FP",), see=("Flag FP",)),
    _e("False negative", "Concept",
       "The model said no but the label says yes. A miss.",
       aliases=("FN",), see=("Flag FN",)),
    _e("Precision", "Validation, Learning",
       "Of the windows the model called the behavior, the share that really were. "
       "Low precision means many false alarms.",),
    _e("Recall", "Validation, Learning",
       "Of the windows that really were the behavior, the share the model found. "
       "Low recall means many misses.",
       aliases=("Sensitivity",),),
    _e("F1", "Validation, Learning",
       "One score that balances precision and recall. 1 is perfect. ABEL reports "
       "it for the behavior itself, not averaged with No Behavior.",
       aliases=("F1 score", "Target-class F1", "Frame F1", "Best Frame F1",
                "Worst Frame F1"),),
    _e("F-beta", "Temporal > Review",
       "Like F1 but weighted. F-beta 1.5 counts misses more heavily than false "
       "alarms.",
       aliases=("Favor recall",),),
    _e("PR-AUC", "Validation, Learning",
       "Area under the precision-recall curve, from 0 to 1. Measures how well the "
       "model ranks true examples above false ones across all thresholds. Suited "
       "to rare behaviors.",
       aliases=("Average precision", "AP"), see=("PR curve",)),
    _e("PR curve", "Learning",
       "A chart of precision against recall as the threshold moves.",
       aliases=("Precision-recall curve",),),
    _e("Confusion matrix", "Learning",
       "A table of what the model called each window against what it really "
       "was. Shows which behaviors get mixed up.",),
    _e("Cross-validation", "Validation",
       "Testing a model several times, each time holding out a different part of "
       "the data, then pooling the results.",
       aliases=("CV", "Fold", "Folds"),),
    _e("Leave-one-mouse-out CV", "Validation > Overview",
       "Trains one model per animal, each time leaving that animal out, and tests "
       "on the left-out animal. Gives one stable score per behavior for how well "
       "models handle new animals. Slow.",
       aliases=("LOSO", "Leave-one-subject-out", "LOMO"),),
    _e("Ablation", "Concept",
       "Removing one part, such as a feature group or a kind of label, and "
       "measuring how much worse the model gets.",
       aliases=("Ablation study",), see=("Label Needs",)),
    _e("Embedding", "Concept",
       "A list of numbers that stands for something complex, such as a video "
       "clip, so similar things get similar numbers.",),
    _e("Supervised", "Concept",
       "Learning from examples you labeled. ABEL's behavior models are supervised.",),
    _e("Unsupervised", "Concept",
       "Finding structure without labels, such as grouping similar windows. "
       "Useful for spotting behaviors you did not plan for.",
       see=("Unsupervised UMAP", "Motif")),
    _e("UMAP", "Learning",
       "A method that squeezes many features into a 2D map where similar windows "
       "land near each other. Used for plots and for picking clips by hand.",
       aliases=("n_neighbors", "Min dist"),
       see=("Select from UMAP", "Unified UMAP")),
    _e("HDBSCAN", "Learning",
       "A clustering method that finds dense groups of points and leaves sparse "
       "points as noise.",
       aliases=("Min cluster size", "Noise (cluster)"),),
    _e("Clustering", "Concept",
       "Grouping similar windows or sessions together without labels.",
       aliases=("Cluster", "K-Means"),),

    # ── Learning subtab ──────────────────────────────────────────────────
    _e("Run Pipeline", "Active Learning > Learning",
       "Full run for one behavior: extract any missing features, build windows, "
       "train, score, pick review clips and evaluate. Use after adding sessions "
       "or changing feature settings.",
       aliases=("Pipeline",),),
    _e("Pipeline All", "Active Learning > Learning",
       "Run Pipeline for every behavior you tick, one after another. Features are "
       "built once and reused.",),
    _e("Retrain", "Active Learning > Learning",
       "Trains the selected behavior again on the saved features with your newest "
       "labels, then scores and picks new clips. Use after a review round. Cannot "
       "be stopped once started.",),
    _e("Retrain All", "Active Learning > Learning",
       "Retrain every behavior you tick, one after another. Each behavior's old "
       "model is replaced.",),
    _e("Run Existing Model", "Active Learning > Learning",
       "Scores this project's data with a saved model as it is, with no training.",),
    _e("Run Models", "Active Learning > Learning",
       "Scores the behaviors you pick with their existing models, without "
       "retraining.",),
    _e("Stop", "Active Learning > Learning",
       "Cancels a running pipeline, Pipeline All or model run. Retrain cannot be "
       "stopped.",
       aliases=("Cancel",),),
    _e("Generate Review Clips", "Active Learning > Learning",
       "When on, batch runs also pick and add review clips for each behavior. "
       "Turn off to only train and score.",),
    _e("Target behavior", "Active Learning > Learning",
       "The behavior the pipeline trains and picks clips for.",),
    _e("Selection mode", "Learning settings",
       "How clips are picked for review. Uncertainty picks clips the model is "
       "unsure about. Random Low-Prob picks random clips the model thinks are "
       "unlikely, which suits a first round.",
       see=("Uncertainty", "Random Low-Prob")),
    _e("Uncertainty", "Learning settings",
       "Picks the windows the model is least sure about. Labeling these teaches "
       "the model the most.",
       aliases=("Uncertainty sampling",),),
    _e("Random Low-Prob", "Learning settings",
       "Picks windows at random among those the model scores low (0.35 or below), "
       "spread over subjects and session time. Good for the first round or to "
       "find missed behavior.",
       aliases=("Random Low-Prob (absent)", "random_absent", "Random windows"),),
    _e("Query size", "Learning settings",
       "How many review candidates each run produces.",),
    _e("Candidate", "Clips, Review",
       "A window the model picked for you to review.",
       aliases=("Candidates", "Review candidate", "Candidate window"),
       see=("Review queue",)),
    _e("Review queue", "Review",
       "The list of candidate clips waiting for your decision.",
       aliases=("Queue",),),
    _e("Candidate source", "Clips",
       "Why a candidate was picked: uncertainty, hard negative, confound "
       "boundary, exploration, mining or a temporal bout.",
       aliases=("Filter Sources", "Source"),),
    _e("Exploration fraction", "Learning settings",
       "Share of the queue filled with random windows, so the model sees areas it "
       "has never been asked about.",
       aliases=("Exploration", "Exploration (random sampling)"),),
    _e("Candidate focus", "Learning settings",
       "Mix of strong predictions and edge cases in the queue. 0% shows only "
       "strong ones. 100% shows only edge cases such as hard negatives.",),
    _e("All-behavior-aware ranking", "Learning settings",
       "Uses the other behaviors' models when picking clips, to find windows "
       "where two behaviors compete.",
       aliases=("Behavior-aware", "Use all-behavior-aware candidate ranking"),),
    _e("Competition margin", "Learning settings",
       "How far the target behavior must beat its competitors for a clip to be "
       "picked. Small values keep close calls.",),
    _e("Weighted queue", "Learning settings",
       "Blends several ranking signals into one score: uncertainty, model "
       "disagreement, diversity, confound boundaries and hard negatives. Most "
       "useful after about 50 labels.",
       aliases=("Enable weighted queue", "Enable weighted queue scoring"),),
    _e("Disagreement", "Learning settings",
       "Picks windows where different behavior models disagree.",),
    _e("Diversity", "Learning settings",
       "Spreads the queue across different kinds of windows so it does not fill "
       "with near copies. Distance to reviewed favors windows unlike any you "
       "labeled.",
       aliases=("Diversity mode", "Distance to reviewed", "Clustering quota"),),
    _e("Confound", "Learning",
       "A behavior that is easy to mix up with another, such as rearing and "
       "climbing. Confound boundary candidates sit between the two.",
       aliases=("Confound-boundary", "confound_boundary", "Confounds"),
       see=("Confound Analysis",)),
    _e("Quick test mode", "Learning settings",
       "Uses a small slice of each session to check that the pipeline works. "
       "Results are not final.",
       aliases=("Enable quick test mode", "Quick Identification", "Very Quick",
                "Quick sample duration", "Legacy preset"),),
    _e("Balanced time sampling", "Learning settings",
       "Uses a random stretch of each session to save time, but runs every step "
       "at full quality.",
       aliases=("Enable balanced time sampling", "Sampling duration", "Sampling seed"),),
    _e("Random seed", "Throughout",
       "A number that fixes the random choices, so the same settings give the "
       "same result. Change it to get a different random draw.",
       aliases=("Seed (random)", "Quick random seed"),),
    _e("Examples per session", "Learning settings",
       "Most candidates picked from any one session. 0 means no cap.",),
    _e("Max segments", "Learning settings",
       "Most windows kept before training. Keeps very large projects fast.",),
    _e("Include imported examples", "Learning settings",
       "Trains on examples imported from other projects as well as this one's.",
       aliases=("Imported data",),),
    _e("Remap reviewed windows", "Learning settings",
       "Moves labels made with an older window size onto the current windows.",
       aliases=("Map reviewed clip IDs to current window length",),),
    _e("Auto-generate features for all reviewed segments", "Learning settings",
       "Computes features for any labeled window that lacks them, so every label "
       "is used in training.",),
    _e("Require GPU", "Learning settings",
       "Stops with an error instead of falling back to the slower CPU.",
       aliases=("Require GPU (fail if fallback)",), see=("GPU",)),
    _e("Skip evaluation reports", "Learning settings",
       "Skips the charts and metrics after training to save time.",
       aliases=("Skip evaluation reports + UMAP (faster)",),),
    _e("Recommend Settings", "Active Learning > Learning",
       "Asks a few questions, looks at your labels and fills in suggested "
       "settings.",
       aliases=("Guided settings",),),
    _e("Rehydrate Settings To UI", "Active Learning > Learning",
       "Loads the settings a saved model was trained with back into the controls.",),
    _e("Model name", "Active Learning > Learning",
       "Optional name for the saved model. Blank uses a timestamp.",
       aliases=("Open Model Folder",),),
    _e("Behavior Separation (UMAP)", "Learning > charts",
       "A 2D map of reviewed windows colored by label. Good separation means the "
       "features tell the behavior apart.",),
    _e("Queue Composition", "Learning > charts",
       "Shows which sources filled the review queue.",),
    _e("Pipeline Timing", "Learning > charts",
       "How long each step of the last run took.",),
    _e("Cross-Behavior Confounds", "Learning > charts",
       "Shows which behaviors the models score high on the same windows.",),
    _e("Unified UMAP", "Learning > charts",
       "One 2D map built from all behavior models, colored by the behavior that "
       "wins each window.",
       aliases=("Unified UMAP (All Behaviors)",),),
    _e("Unsupervised UMAP", "Learning > charts",
       "A 2D map of the raw features with no models or labels, colored by "
       "automatically found clusters.",
       aliases=("Unsupervised UMAP (Clusters)",),),
    _e("Expert Assignment", "Learning > charts",
       "Shows which saved model is assigned to each behavior.",
       aliases=("Expert Assignment (Per Model)",),),
    _e("Select from UMAP", "Active Learning > Learning",
       "Opens a 2D map where you can lasso a group of windows and send them to "
       "clip extraction.",
       aliases=("UMAP Interactive Selection", "Lasso"),),
    _e("Confound Analysis", "Active Learning > Learning",
       "A table of which behaviors overlap, with suggestions for better labels.",),
    _e("Find Edge Cases", "Active Learning > Learning",
       "Finds windows where two behaviors score close to each other, so you can "
       "review them.",
       aliases=("Find Competing Windows",),),
    _e("Test Behavior Awareness", "Active Learning > Learning",
       "A quick test that compares results with and without the other behaviors' "
       "models taken into account.",),

    # ── Seeds subtab ─────────────────────────────────────────────────────
    _e("Mark Start", "Active Learning > Seeds",
       "Sets the current video frame as the start of a seed. Mark End sets its "
       "end.",
       aliases=("Mark End", "Mark Start (current frame)"),),
    _e("Add Seed", "Active Learning > Seeds",
       "Saves the marked frame range as an example of the selected behaviors.",
       aliases=("Mark Seed Bout",),),
    _e("Copy Seed to Other Subjects", "Active Learning > Seeds",
       "Copies a seed's behavior and frame range to other subjects. Use it when a "
       "behavior happens at the same time in every video.",),
    _e("Relink Orphaned Seeds", "Active Learning > Seeds",
       "Reconnects seeds whose session ids no longer exist to the right sessions.",),
    _e("Assume All Non-Positive Frames = Negative", "Active Learning > Seeds",
       "Treats every frame outside your seeds as a negative, so you only have to "
       "mark the positive bouts.",),
    _e("Delete Selected Seed", "Active Learning > Seeds",
       "Removes the selected seed.",),

    # ── Clips subtab ─────────────────────────────────────────────────────
    _e("Clip", "Clips, Review",
       "A short video file cut around one candidate window, for review.",
       aliases=("Review clip", "Clips"),),
    _e("Extract Clips", "Active Learning > Clips",
       "Cuts video clips for the candidates in the table.",),
    _e("Top / bottom / median N", "Active Learning > Clips",
       "How many of the highest, lowest and middle scoring candidates to cut per "
       "subject.",
       aliases=("Top candidates", "Top N", "Bottom N", "Median N"),),
    _e("Padding", "Active Learning > Clips",
       "Seconds of video added before and after each clip for context.",
       aliases=("Padding (s)", "Before", "After"),),
    _e("Clip preset", "Active Learning > Clips",
       "Clip size and crop settings for extracted clips.",
       aliases=("Preset (clips)",),),
    _e("Crop area", "Active Learning > Clips",
       "How large the crop around the animal is, relative to the preset.",
       aliases=("Crop margin", "Default crop margin"),),
    _e("Keep all animals in view", "Active Learning > Clips",
       "In multi-animal sessions, widens the crop so every animal stays in the "
       "clip.",),
    _e("Full frame (no crop)", "Active Learning > Clips",
       "Writes the whole video frame instead of a crop. Shows the whole arena but "
       "looks softer.",
       aliases=("Full frame",),),
    _e("ID marker size", "Active Learning > Clips",
       "Size of the colored dot drawn on each animal in multi-animal clips.",
       aliases=("ID marker", "Identity marker"),),
    _e("Clear Clips", "Active Learning > Clips",
       "Deletes the clip video files for the selected subject to free disk "
       "space. Labels and the queue are kept.",),
    _e("Clear Candidates", "Active Learning > Clips",
       "Removes unreviewed candidates from the queue. Reviewed ones and their "
       "labels are kept.",),
    _e("Hide reviewed clips", "Active Learning > Clips",
       "Hides candidates you already decided on.",
       aliases=("Hide reviewed",),),

    # ── Review subtab ────────────────────────────────────────────────────
    _e("Accept", "Active Learning > Review",
       "The clip shows the behavior in Review label. Saves it as a positive.",
       aliases=("Accepted",),),
    _e("Reject", "Active Learning > Review",
       "The clip does not show the behavior it was picked for. Saves it as a "
       "negative for that behavior only.",
       aliases=("Rejected",),),
    _e("Relabel", "Active Learning > Review",
       "The clip shows a different behavior. Saves it under the behavior chosen "
       "in Review label.",),
    _e("Boundary Error", "Active Learning > Review",
       "The behavior is in the clip but starts or ends at the wrong place. Noted "
       "for your records and not used in training.",),
    _e("Skip", "Active Learning > Review",
       "Moves on without saving a label. Use it for clips you cannot judge.",),
    _e("Unsure", "Validation Quiz",
       "Answer for a quiz clip you cannot judge. Unsure answers are left out of "
       "the scores and out of training.",),
    _e("Review label", "Active Learning > Review",
       "The behavior that Accept, Relabel and Add Label apply.",
       aliases=("Add Label", "Clear Labels for This Clip"),),
    _e("Commit Labels for This Clip", "Active Learning > Review",
       "Saves all labels staged for the clip. Shortcut: Enter.",
       aliases=("Save Decision", "Commit", "Commit Labels to Training"),),
    _e("Behavior Soundboard", "Active Learning > Review",
       "A pop-out with one button per behavior for labeling the current clip. In "
       "multi-animal clips it labels each animal separately.",
       aliases=("Soundboard",),),
    _e("Auto-advance", "Review, Validation Quiz",
       "Moves to the next clip after you save a decision.",
       aliases=("Auto-advance on save", "Auto-advance by default"),),
    _e("Loop", "Review",
       "Replays the clip from the start when it ends.",
       aliases=("Loop clips by default",),),
    _e("Playback speed", "Review",
       "Plays clips faster or slower.",
       aliases=("Speed",),),
    _e("Show reviewed candidates", "Active Learning > Review",
       "Lists clips you already decided on, plus your seeds, so you can check or "
       "change them.",
       aliases=("Show reviewed", "Reviewed + Clips Only", "Skip clips I've already reviewed"),),
    _e("Temporal Bout Review Clips", "Active Learning > Review",
       "Shows only clips sent over from detected bouts in Temporal > Review.",),
    _e("Active Learning FP / FN", "Active Learning > Review",
       "Shows labeled clips the current model now disagrees with: negatives it "
       "scores high and positives it scores low. They may be mislabeled.",),
    _e("Flag Outliers (Dissimilarity)", "Active Learning > Review",
       "Highlights accepted clips that look unlike the rest of their behavior. "
       "They may need a second look.",),
    _e("Regenerate Missing Clips", "Active Learning > Review",
       "Cuts clip files again for candidates whose clips are missing.",),
    _e("Remove Selected", "Active Learning > Review",
       "Deletes the selected candidates, their clips and any saved decisions.",),
    _e("Clear Labels", "Active Learning > Review",
       "Erases saved decisions for the selected clips and puts them back in the "
       "queue. The clips are kept.",),
    _e("Start / End frame", "Active Learning > Review",
       "The frame range stored with a clip's decision. Edit it when the behavior "
       "starts or ends somewhere else in the clip, then save the decision. Reset "
       "puts back the clip's own range.",
       aliases=("Apply Frame Range", "Start frame", "End frame"),),
    _e("Bulk Assign", "Active Learning > Review",
       "Applies a behavior to many clips at once: the selected rows, or the whole "
       "visible queue.",
       aliases=("Accept All", "Assign"),),

    # ── Targeted Clip Mining ─────────────────────────────────────────────
    _e("Targeted Clip Mining", "Active Learning > Review",
       "Finds clips by what the animal is doing, such as nose past the zone edge "
       "or high speed. Useful for rare behaviors.",
       aliases=("Clip Mining", "Mining", "Clear Mining Filter"),
       see=("Extract Essence", "Auto Hunter", "Label Needs")),
    _e("Criteria", "Targeted Clip Mining",
       "The feature ranges a window must match. Add, edit or turn off each one.",
       aliases=("Criterion",),),
    _e("Find Matches", "Targeted Clip Mining",
       "Scores every window against the criteria and counts the matches.",),
    _e("Extract Essence", "Targeted Clip Mining",
       "Select a few good example clips first. ABEL finds the features that set "
       "them apart from all other windows, adds them as criteria, and ranks every "
       "window by how much it resembles your examples.",
       aliases=("Essence", "Essence Extractor", "Extract Essence from Selected Clips"),),
    _e("Load Matches into Review Queue", "Targeted Clip Mining",
       "Cuts clips for the best matches and adds them to the review queue.",),
    _e("Flag Failing Clips in Review Queue", "Targeted Clip Mining",
       "Highlights clips in the review list that fall outside the current "
       "criteria. Catches clips that no longer fit a tighter definition.",),
    _e("Spread evenly across subjects", "Targeted Clip Mining",
       "Fills the batch by taking turns between animals, so one animal does not "
       "fill it.",),
    _e("Auto Hunter", "Targeted Clip Mining",
       "Ready-made searches for common behaviors that work with nothing reviewed "
       "yet.",
       aliases=("Use This Preset", "Hunt preset"),),
    _e("Label Needs", "Targeted Clip Mining",
       "Tests what kind of new label would help each model most: more positives, "
       "positives from more animals, or more negatives. Changes nothing in the "
       "project.",
       aliases=("Run Label Needs Test",),),
    _e("Coverage gap", "Targeted Clip Mining",
       "An animal with few or no labeled examples of a behavior. Coverage-gap "
       "mining looks for clips only in those animals.",
       aliases=("Only subjects lacking examples of", "Hunt Coverage Gaps for Selected"),),

    # ── Temporal ─────────────────────────────────────────────────────────
    _e("Temporal refinement", "Temporal > Refinement",
       "Scores windows every few frames, averages the overlapping scores into one "
       "probability per frame, then turns those traces into bouts with start and "
       "end frames.",
       aliases=("Dense inference",), see=("Probability trace", "Bout")),
    _e("Select Behavior Models", "Temporal > Refinement",
       "Choose which saved model scores each behavior. Exclude from competition "
       "keeps a behavior, usually No Behavior, out of inhibition.",
       aliases=("Exclude from competition", "Auto (latest for this behavior)"),),
    _e("Inference step", "Temporal > Refinement",
       "Seconds between scored windows during refinement. Smaller gives finer "
       "timing but takes longer. 0.1 s is the default.",
       aliases=("Inference step (seconds)",),),
    _e("Warmup suppress", "Temporal > Refinement",
       "Ignores predictions in the first seconds of each session, when the animal "
       "has just been placed and scores are unreliable.",
       aliases=("Warmup", "Warmup suppress (seconds)"),),
    _e("Inhibition weight", "Temporal > Refinement",
       "Lets behaviors compete. Each behavior's probability is lowered by this "
       "weight times the others' probabilities, so the strongest one stands out. "
       "0 turns it off.",
       aliases=("Mutual inhibition", "Inhibition", "Apply inhibition"),
       see=("Configure Suppression",)),
    _e("Configure Suppression", "Temporal > Refinement",
       "Set inhibition for specific pairs of behaviors and preview the effect.",
       aliases=("Suppression matrix", "Suppression Helper", "Make Symmetric",
                "Fill Uniform", "Reset All to 0"),),
    _e("Parallel inference", "Temporal > Refinement",
       "Scores several sessions at once. Turn off if memory runs low.",),
    _e("Test mode (1 session)", "Temporal > Refinement",
       "Runs on one session only to check settings quickly.",),
    _e("Clear Temporal Cache", "Temporal > Refinement",
       "Deletes saved refinement results so the next run starts fresh.",
       aliases=("Clear cache first (temporal)",),),
    _e("Probability trace", "Temporal",
       "A line showing the probability of one behavior across every frame of a "
       "session.",
       aliases=("Trace", "Behavior Probability Traces", "Confidence trace"),),
    _e("Winning Behavior Timeline", "Temporal > Refinement",
       "A strip showing which behavior has the highest probability at each frame.",),
    _e("Behavior Probability Heatmap", "Temporal > Refinement",
       "All behaviors' probabilities over time as colored rows.",),
    _e("Bout", "Temporal, Analytics",
       "One continuous stretch of a behavior, with a start and end frame.",
       aliases=("Bouts", "Behavior bout"),
       see=("Min bout", "Merge gap")),
    _e("Min bout", "Temporal > Review",
       "Shortest stretch that counts as a bout. Shorter ones are dropped.",
       aliases=("Minimum bout duration", "Min bout length"),),
    _e("Merge gap", "Temporal > Review",
       "Bouts closer together than this are joined into one.",),
    _e("Per-Behavior Thresholds", "Temporal > Review",
       "Sets threshold, minimum bout and merge gap for each behavior.",
       see=("Threshold", "Min bout", "Merge gap")),
    _e("Auto-Tune Settings", "Temporal > Review",
       "Suggests each behavior's threshold from its held-out predictions, plus "
       "min bout and merge gap. Fills the table only. Nothing is saved until you "
       "apply.",
       aliases=("Auto-Tune",),),
    _e("Apply + Process All Behaviors", "Temporal > Review",
       "Saves the thresholds and rebuilds the bouts for every behavior.",),
    _e("Flag FP", "Temporal > Review",
       "Click a detected bout on the trace to mark it as a false alarm. Right-click "
       "to undo.",
       aliases=("Flag FP (click bout)", "Add False Positive Interval"),),
    _e("Flag FN", "Temporal > Review",
       "Drag across the trace to mark a stretch the model missed. Right-click to "
       "undo.",
       aliases=("Flag FN (drag range)", "Add False Negative Interval"),),
    _e("Commit Flags", "Temporal > Review",
       "Saves the staged FP and FN flags as labels. Until then they are only "
       "drawn on screen.",
       aliases=("Clear Staged",),),
    _e("Send All Bouts to Clip Review", "Temporal > Review",
       "Turns every detected bout into review clips and sends them to Active "
       "Learning > Clips.",
       aliases=("Send Current Behavior/Subject Bouts", "Send to Clip Extraction"),),
    _e("Seek Bout", "Temporal > Review",
       "Jumps the video to the start of the selected bout.",),
    _e("Session Quality", "Validation > Overview",
       "Flags sessions whose results look odd compared with other sessions of the "
       "same type, such as very low confidence or unusual bout counts.",
       aliases=("Session Quality…", "Z-threshold", "Anomaly"),),
    _e("Bout IoU", "Temporal > Review",
       "Overlap between a detected bout and a labeled one, from 0 (none) to 1 "
       "(exact).",
       aliases=("IoU", "Intersection over union", "Best Bout IoU", "Worst Bout IoU"),),
    _e("Onset error", "Temporal > Review",
       "How far a detected bout's start is from the labeled start. Offset error "
       "is the same for the end.",
       aliases=("Offset error", "Highest Onset Error", "Highest Offset Error"),),

    # ── Analytics ────────────────────────────────────────────────────────
    _e("Refresh Analytics", "Analytics",
       "Reloads bouts and recomputes the tables and graphs.",),
    _e("Factor", "Analytics > Summary",
       "A grouping column such as Sex, Treatment or Genotype. Each session gets a "
       "level for each factor.",
       aliases=("Add Factor", "Remove Factor", "Group factor"),
       see=("Level",)),
    _e("Level", "Analytics > Summary",
       "One value of a factor, such as Male or Saline.",
       aliases=("Level Order", "Assign Factor to Selected"),),
    _e("Auto-Group by Session Type", "Analytics > Summary",
       "Makes a factor from each session's type, such as Conditioning or Recall.",),
    _e("Merge Projects", "Analytics",
       "Brings bouts from other ABEL projects in, so you can analyze cohorts "
       "together.",
       aliases=("Combined projects", "Add Project", "Include combined projects"),),
    _e("Per-Subject Prechop", "Analytics",
       "Sets a start offset per subject, so analysis begins after an initial "
       "period you want to skip.",
       aliases=("Prechop",),),
    _e("ROI scope", "Analytics",
       "Limits every analysis to behavior inside or outside one zone. Choose how "
       "bouts that cross the zone edge are counted.",
       aliases=("Whole arena", "Split at the boundary"),),
    _e("Animal (analytics)", "Analytics",
       "In multi-animal sessions, which animal to analyze. Either animal counts a "
       "behavior when either animal does it.",
       aliases=("Either animal", "Any animal"),),
    _e("Bout Count", "Analytics",
       "Number of bouts of a behavior.",
       aliases=("n_bouts",),),
    _e("Total Duration", "Analytics",
       "Total time spent in a behavior.",
       aliases=("Time spent", "Duration"),),
    _e("Mean Bout Duration", "Analytics",
       "Average length of one bout.",),
    _e("Latency", "Analytics",
       "Time from the start of the session to the first bout of a behavior.",
       aliases=("Latency to First",),),
    _e("Distance Traveled", "Analytics",
       "How far the animal moved, in cm when px/mm is set.",
       aliases=("Distance",),),
    _e("Time bin", "Analytics > Graphs",
       "Bin size in seconds for time-course charts.",
       aliases=("Bin", "Time-bin"),),
    _e("Data Range", "Analytics > Graphs",
       "Limits analysis to bouts between a start and end time. 600 s is 10 "
       "minutes.",
       aliases=("Data range from", "Start time bins at Data Range 'from'"),),
    _e("Bout filter", "Analytics > Graphs",
       "First N Bouts keeps only each session's first bouts. Bouts Until Behavior "
       "keeps bouts before the first bout of another behavior.",
       aliases=("First N Bouts", "Bouts Until Behavior", "Scale by pre-behavior time"),),
    _e("Combine / split", "Analytics",
       "For each factor: combine pools across it, split draws one series per "
       "level, or pick one level to keep only it.",
       aliases=("Facet",),),
    _e("Statistics", "Analytics",
       "Runs a t test or ANOVA on group data, and lets you choose paired or "
       "unpaired, parametric or rank-based, and the follow-up test.",
       aliases=("Run Statistics", "Statistics…", "View Stats", "Stats"),
       see=("P value", "Post-hoc test")),
    _e("Paired", "Statistics options",
       "Compares each subject with itself across groups, such as the same mouse "
       "on Day 1 and Day 2.",
       aliases=("Unpaired", "Repeated measures"),),
    _e("Parametric test", "Statistics options",
       "Assumes the data are roughly bell-shaped (Gaussian). When off, ABEL uses "
       "rank-based tests that do not assume this.",
       aliases=("Assume Gaussian distribution", "Gaussian", "Rank-based test",
                "Nonparametric"),),
    _e("t test", "Statistics",
       "Compares the means of two groups.",
       aliases=("Student's t test", "Welch's t test", "Paired t test"),),
    _e("ANOVA", "Statistics",
       "Compares the means of three or more groups, or two factors at once "
       "(two-way).",
       aliases=("One-way ANOVA", "Two-way ANOVA", "Welch's ANOVA"),),
    _e("Rank tests", "Statistics",
       "Mann-Whitney, Wilcoxon, Kruskal-Wallis and Friedman. Compare groups by "
       "rank instead of raw value.",
       aliases=("Mann-Whitney", "Wilcoxon", "Kruskal-Wallis", "Friedman"),),
    _e("Welch correction", "Statistics options",
       "Allows groups to have different spreads. On when Assume equal SDs is off.",
       aliases=("Welch", "Assume equal SDs"),),
    _e("Geisser-Greenhouse correction", "Statistics options",
       "Corrects repeated-measures tests when the spread of differences is "
       "unequal. Prism's default.",
       aliases=("Sphericity",),),
    _e("Post-hoc test", "Statistics options",
       "Pairwise comparisons after an overall test. Tukey compares every pair. "
       "Dunnett compares each group with a control. Sidak, Holm-Sidak and "
       "Bonferroni correct for many tests. Fisher's LSD does not correct.",
       aliases=("Tukey", "Dunnett", "Sidak", "Holm-Sidak", "Bonferroni",
                "Fisher's LSD", "Dunn", "Games-Howell", "Multiple comparisons",
                "Only run comparisons when the overall test is significant"),),
    _e("P value", "Statistics",
       "The chance of a difference this large if the groups were really the same. "
       "Below 0.05 is usually called significant. Stars on graphs mark it.",
       aliases=("p", "Significance", "Show statistics on graph"),),
    _e("Error bars", "Analytics > Graph settings",
       "SEM shows uncertainty in the mean. SD shows spread of the data. 95% CI is "
       "the range likely to hold the true mean.",
       aliases=("SEM", "SD", "95% CI", "Standard error", "Standard deviation",
                "Confidence interval"),),
    _e("Show individual data points", "Analytics > Graph settings",
       "Draws each subject's value on top of the bars.",),
    _e("Copy for Prism", "Analytics",
       "Copies the graph's values as columns you can paste into GraphPad Prism.",
       aliases=("Prism",),),
    _e("Graph Size", "Analytics, Learning",
       "Sets the largest width and height of the graph area.",
       aliases=("Force fit to canvas size",),),
    _e("Groups (graphs)", "Analytics > Graphs",
       "Sets the order and colors of groups.",
       aliases=("Groups…", "Colors"),),
    _e("Export Figure", "Analytics",
       "Saves the current graph as PNG, SVG or PDF.",
       aliases=("Export PNG", "Export Vector (SVG/PDF)", "Save Visualization",
                "Export Chart"),),
    _e("Export Data", "Analytics",
       "Saves the numbers behind the current graph as CSV or Excel.",
       aliases=("Export CSV (graph)", "Export Excel", "Export to Excel",
                "Export Profile Data"),),
    _e("Bandwidth", "Analytics > Spatial Heatmap, Density Analysis",
       "How far each point is spread when drawing a heatmap. Smaller gives "
       "tighter spots, larger gives smoother maps.",
       aliases=("KDE", "Kernel density", "Auto-detect bandwidth"),),
    _e("Clean plate", "Analytics > Spatial Heatmap",
       "A picture of the empty arena built from the video with the animal "
       "removed. Used as the heatmap background.",
       aliases=("Regenerate BG", "Background Image", "Image Adjust"),),
    _e("Generate Heatmap", "Analytics > Spatial Heatmap",
       "Draws the spatial heatmap with the current settings.",
       aliases=("Generate Density Map", "Generate Comparison"),),
    _e("Difference map", "Analytics > Density Analysis > Group Comparison",
       "Shows where group A spends more or less time than group B. Signed is A "
       "minus B. Normalized divides by the total. Log2 ratio shows how many times "
       "more.",
       aliases=("Signed difference", "Log2 ratio", "Normalized diff", "Mask"),),
    _e("Transition matrix", "Analytics > Behavior Relationships",
       "How often each behavior is followed by each other behavior. With "
       "Probabilities ticked, each row is shown as shares that add to 1 instead "
       "of raw counts.",
       aliases=("Transition", "Transitions", "Delta heatmap", "Network / chord",
                "Probabilities (transitions)", "Row-normalized"),
       see=("Max gap", "Permutation test")),
    _e("Max gap", "Analytics > Behavior Relationships",
       "Longest time between the end of bout A and start of bout B for A to B to "
       "count as a transition.",
       aliases=("Gap (transitions)",),),
    _e("Overlap tolerance", "Analytics > Behavior Relationships > settings",
       "How far bout B may start before bout A ends and still count as A then B. "
       "Bouts that overlap more are treated as happening at once.",),
    _e("Include self-transitions", "Analytics > Behavior Relationships",
       "Counts a behavior followed by itself.",
       aliases=("Self-transition",),),
    _e("Permutation test", "Analytics > Behavior Relationships",
       "Tests group differences by shuffling group labels many times and seeing "
       "how often the shuffled difference is as large as the real one.",),
    _e("Motif", "Analytics > Behavior Relationships",
       "A short sequence of behaviors that repeats, such as rear then groom then "
       "rear.",
       aliases=("Motifs",),
       see=("N-gram",)),
    _e("N-gram", "Analytics > Behavior Relationships",
       "A sequence of N behaviors in a row. N-gram frequency counts the most "
       "common ones.",
       aliases=("N-gram Frequency", "Skip consecutive same"),),
    _e("Sequence Clustering", "Analytics > Behavior Relationships",
       "Groups sessions with similar motif patterns.",),
    _e("Engram sequence view", "Analytics > Behavior Relationships",
       "Shows motifs as colored blocks with how common each is per group.",),
    _e("HMM", "Analytics > Behavior Relationships > HMM Analysis",
       "Hidden Markov model. Treats behavior as moving between a few hidden "
       "states, each with its own mix of behaviors, and finds those states from "
       "the bout sequence.",
       aliases=("Hidden Markov model", "HMM Analysis", "Run HMM", "Hidden state",
                "State"),
       see=("Viterbi", "Auto-calibrate")),
    _e("Emission", "HMM Analysis",
       "What each hidden state produces: which behaviors are likely while in that "
       "state.",
       aliases=("Emission heatmap",),),
    _e("Viterbi", "HMM Analysis",
       "Assigns every bout to the single most likely state path. Each bout belongs "
       "to one state.",
       aliases=("Viterbi (hard)", "Viterbi state bouts"),),
    _e("Posterior", "HMM Analysis",
       "The probability of each state for each bout. A bout two states could "
       "explain is split between them.",
       aliases=("Posterior (soft)", "Top-N posterior"),),
    _e("Auto-calibrate", "HMM Analysis",
       "Tests HMM settings on your data and proposes how many states to use, with "
       "the evidence. Nothing changes until you approve.",
       aliases=("HMM calibration",),),
    _e("AIC / BIC", "HMM Analysis",
       "Scores that balance how well a model fits against how complex it is. "
       "Lower is better. ICL also penalizes states that overlap. AICc corrects "
       "AIC for small data.",
       aliases=("AIC", "BIC", "AICc", "ICL", "Criterion"),),
    _e("EM restarts", "HMM Analysis",
       "Number of times the HMM fit starts from a different random guess. The "
       "best fit is kept. More restarts avoid a poor fit.",
       aliases=("EM", "Restarts", "Expectation maximization"),),
    _e("Min dwell", "HMM Analysis",
       "Shortest time in a state that counts as entering it, so a brief flicker "
       "does not set the latency.",),
    _e("State occupancy", "HMM Analysis",
       "Share of a subject's time spent in each hidden state.",),
    _e("Ethogram", "HMM Analysis, Export",
       "A timeline with one row per subject, colored by behavior or state over "
       "time.",
       aliases=("State ethogram", "Export Ethogram CSV"),),
    _e("Export State Boutframes", "HMM Analysis",
       "Writes hidden states as a boutframes workbook, the same shape as the "
       "behavior one, so TRACY can align states to photometry.",
       aliases=("HMM_NoState",),),
    _e("ITI", "Analytics > Session Sections",
       "Inter-trial interval. The gap between trials, such as between tones in "
       "fear conditioning.",),
    _e("Section preset", "Analytics > Session Sections",
       "A saved list of named time blocks for a session design.",
       aliases=("Save as Custom", "Load Saved", "Add Section",
                "Average repeated selected sections"),),
    _e("Velocity profile", "Analytics > Velocity",
       "Average speed across each bout, scaled so 0% is bout start and 100% is "
       "bout end, with time before and after shown.",
       aliases=("Pre/post context", "First vs Last"),),
    _e("Outlier Detection", "Analytics > Velocity",
       "Finds bouts with unusual speed. IQR uses the middle half of the data. "
       "Z-score uses distance from the mean. Exclude drops them, Winsorize caps "
       "them.",
       aliases=("Outlier", "IQR", "Z-score", "Winsorize"),),
    _e("Dyad", "Analytics > Social Interaction",
       "A pair of animals recorded together.",
       aliases=("Dyadic",),),
    _e("Displacement", "Analytics > Social Interaction",
       "One animal moves toward the other while the other moves away, both faster "
       "than a set speed.",
       aliases=("Displacement threshold",),),
    _e("Dominance HMM", "Analytics > Social Interaction",
       "Fits hidden interaction states to social and movement features and ranks "
       "animals by how often they displace the other.",
       aliases=("Run Dominance HMM", "Dominance", "Dominance (HMM)"),),
    _e("Compute Social Metrics", "Analytics > Social Interaction",
       "Loads the social features and summarizes each pair of animals.",),

    # ── Validation ───────────────────────────────────────────────────────
    _e("Generate Test", "Validation > Validation Quiz",
       "Builds a new random quiz from the current settings.",
       aliases=("Test Settings",),),
    _e("Resume Saved Test", "Validation > Validation Quiz",
       "Reopens your most recent quiz.",
       aliases=("Delete Test",),),
    _e("Reviewer", "Validation > Validation Quiz",
       "The person taking the quiz. Answers are stored per reviewer name.",
       aliases=("Reviewer name",),),
    _e("Write-back reviewer", "Validation > Validation Quiz",
       "Copies a reviewer's quiz answers, except Unsure, into the training labels.",),
    _e("Cohen's kappa", "Validation",
       "Agreement between two raters, corrected for agreement by chance. 1 is "
       "perfect, 0 is chance. Fleiss' kappa is the same for more than two raters.",
       aliases=("Kappa", "Fleiss' kappa", "Inter-rater agreement"),),
    _e("Intra-rater reliability", "Validation",
       "How consistently one reviewer labels the same clips a second time.",
       aliases=("Intra-rater reviewer",),),
    _e("Strongest bouts", "Validation > Behavior Grid",
       "Fills the grid with the most confident detections.",),
    _e("Probability bands", "Validation > Behavior Grid",
       "Each grid row shows a different probability range, from highest at top to "
       "lowest accepted at bottom.",),
    _e("Measure Review Effort", "Validation > Review Effort",
       "Computes labeling time from decision timestamps. Gaps longer than the "
       "break cutoff are left out.",
       aliases=("Break cutoff", "Batch gap", "Adapt"),),

    # ── Export ───────────────────────────────────────────────────────────
    _e("Export CSV", "Export",
       "Writes reviewed clips and their labels to a spreadsheet file.",),
    _e("Export Word Behavior Matrix", "Export",
       "Writes a Word file with one page per behavior and a 0/1 table of whether "
       "it was present in each frame for each subject.",),
    _e("Boutframes workbook", "Export",
       "An Excel file with one sheet per subject and the start frame of every "
       "bout per behavior, in raw video frames. TRACY reads it to line bouts up "
       "with photometry.",
       aliases=("Export Boutframes Workbook", "Boutframes", "Include end-frame columns"),),
    _e("Binary frame labels (0/1)", "Export",
       "Writes one row per frame with a 0 or 1 per behavior instead of bout start "
       "lists.",),
    _e("TRACY", "Export",
       "A separate program for fiber photometry analysis that reads ABEL's "
       "boutframes and position files.",
       aliases=("Photometry", "Fiber photometry"),),
    _e("Export TRACY Position Files", "Export",
       "Writes one position file per session (the animal's x, y track) for TRACY.",
       aliases=("ABELposition", "Track point"),),
    _e("Export Labeled Tracking Videos", "Export",
       "Writes videos with the behavior labels and tracking drawn on them.",
       aliases=("Labeled video", "Customize Overlay", "Overlay"),),
    _e("View Behaviogram", "Export",
       "Shows a timeline of accepted behavior clips per subject.",
       aliases=("Behaviogram",),),

    # ── Direct Use and model reuse ───────────────────────────────────────
    _e("Workflow snapshot", "Direct Use",
       "The source project's models, window settings, thresholds and bout "
       "settings, read fresh each time Direct Use runs so new videos are scored "
       "exactly as the source project scores its own.",
       aliases=("Snapshot", "Create Snapshot Workflow"),),
    _e("Run Direct Use Pipeline", "Direct Use > Run",
       "Runs the snapshot on the new videos: features, scoring, refinement and "
       "bouts.",
       aliases=("Direct Use Workflow", "Select Direct Use Project"),),
    _e("Source project", "Direct Use, Model Refinement",
       "The project whose models or examples you are reusing.",
       aliases=("Add Source Project", "Select Source Project"),),
    _e("Import Selected Models", "Direct Use > Apply Models",
       "Copies the chosen models from the source project into this one.",),
    _e("Analyze Transfer", "Direct Use > Transfer Feedback",
       "Runs the transfer checks and ranks subjects worst first.",),
    _e("Import Project as Baseline", "Model Refinement",
       "Seeds this project from another one's clips, labeled features and models.",
       aliases=("Baseline",),),
    _e("Import Compatible Examples", "Model Refinement",
       "Adds the source project's labeled examples whose features match this "
       "project.",),
    _e("Map Behaviors", "Model Refinement",
       "Matches behaviors that have different names but mean the same thing, "
       "such as Head Dip and Dip, so their examples import.",
       aliases=("Map Behavior Names",),),
    _e("Remove Imported", "Model Refinement",
       "Un-imports a project: deletes its imported rows, clips and review entries.",
       aliases=("Remove Import",),),
    _e("Diagnose Models", "Model Refinement",
       "Explains why imported models are not compatible and how to fix it.",
       aliases=("Diagnose Models: Why and How to Fix",),),
    _e("Feature-value shift", "Model Refinement",
       "How different the source project's feature values are from this one's. "
       "Large shifts make imported examples less useful.",),

    # ── Settings and app ─────────────────────────────────────────────────
    _e("Autosave seconds", "Settings > Preferences",
       "How often ABEL saves your work automatically.",
       aliases=("Max recent projects", "Check updates on startup", "Save Settings"),),
    _e("GPU", "Settings > Dependencies",
       "A graphics card. Training, optical flow and R3D run much faster on an "
       "NVIDIA GPU. ABEL falls back to the CPU when there is none.",
       aliases=("CUDA", "CPU", "Graphics card"),),
    _e("Install GPU PyTorch", "Settings > Dependencies",
       "Finds your NVIDIA card and driver, then installs and checks the matching "
       "PyTorch build.",
       aliases=("PyTorch", "torch"),),
    _e("Install All Dependencies", "Settings > Dependencies",
       "Installs every missing Python package ABEL needs.",
       aliases=("Uninstall Selected",),),
    _e("Run Smoke Test", "Settings > Dependencies",
       "Pushes made-up data through the CPU and GPU code paths and reports what "
       "passed and how long it took.",),
    _e("Copy Diagnostic Report", "Settings > Dependencies",
       "Copies package versions and system details to send with a bug report.",),
    _e("Check for Updates", "Info > About & Updates",
       "Asks GitHub whether a newer version exists. ABEL never updates on its own.",),
    _e("Install Update", "Info > About & Updates",
       "Downloads the newest version and restarts ABEL. Unsaved work is lost.",),

    # ── Files ────────────────────────────────────────────────────────────
    _e("derived folder", "Project folder",
       "Where ABEL keeps what it computes: features, models, labels, bouts and "
       "evaluation results.",
       aliases=("derived/", "exports/"),),
    _e("Parquet", "Project folder",
       "A compact table file format ABEL uses for features, labels and bouts.",
       aliases=(".parquet",),),
)


_BY_TERM: dict[str, GlossaryEntry] = {e.term: e for e in ENTRIES}


def entry(term: str) -> GlossaryEntry | None:
    return _BY_TERM.get(term)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


def _squash(s: str) -> str:
    return _norm(s).replace(" ", "")


def search(query: str) -> list[GlossaryEntry]:
    """Entries matching *query*, best match first. An empty query returns all A-Z."""
    q = _norm(query)
    if not q:
        return sorted(ENTRIES, key=lambda e: e.term.lower())
    q_sq = q.replace(" ", "")
    words = q.split()
    scored: list[tuple[int, str, GlossaryEntry]] = []
    for e in ENTRIES:
        names = [e.term, *e.aliases]
        norm_names = [_norm(n) for n in names]
        sq_names = [_squash(n) for n in names]
        if q in norm_names[:1] or q_sq == sq_names[0]:
            score = 100
        elif q in norm_names or q_sq in sq_names:
            score = 90
        elif norm_names[0].startswith(q):
            score = 80
        elif any(n.startswith(q) for n in norm_names):
            score = 70
        elif any(f" {q} " in f" {n} " for n in norm_names):
            score = 60
        elif any(q_sq in n for n in sq_names):
            score = 50
        else:
            body = f" {_norm(e.text)} {_norm(e.where)} "
            if all(f" {w}" in body for w in words):
                score = 20 if all(f" {w} " in body for w in words) else 10
            else:
                continue
        scored.append((score, e.term.lower(), e))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in scored]
