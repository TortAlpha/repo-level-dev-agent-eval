import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WORK = path.join(ROOT, 'presentation');
const BUILD = path.join(WORK, '.build');
const SKILL = process.env.PRESENTATIONS_SKILL_DIR;
if (!SKILL) throw new Error('Set PRESENTATIONS_SKILL_DIR to the installed Presentations skill directory.');
const PYTHON = process.env.PYTHON_EXECUTABLE ?? 'python3';
if (!process.env.RUNTIME_NODE_MODULES) throw new Error('Set RUNTIME_NODE_MODULES to the runtime node_modules directory.');
const {resolvePresentationFont, applyPresentationChartFont, finalizePresentation} = await import(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs'));
const FONT = 'Helvetica Neue';
const REFERENCE = path.join(WORK, 'output', 'Repository_Level_Agent_Evaluation_Defense_EN_v4.pptx');
const REFERENCE_SHA = createHash('sha256').update(await fs.readFile(REFERENCE)).digest('hex');
const REPO = path.resolve(ROOT, '../..');
const REPEAT_SOURCE = path.join(REPO, 'experiments/plans/defense_repeats_20260904/analysis');
const RA = JSON.parse(await fs.readFile(path.join(REPEAT_SOURCE, 'audit_repetitions.json'), 'utf8'));
const RM = ['single','multi-graph'];
const RF = r => `${r.numerator}/${r.denominator}`;
const RP = r => `${(100*r.rate).toFixed(1)}% (${RF(r)})`;
const USD = n => `$${n.toFixed(4)}`;
const RG = RA.architectures;
const repeatSources = [path.join(REPEAT_SOURCE,'audit_repetitions.json'),path.join(REPEAT_SOURCE,'provenance_review.md'),path.join(REPO,'experiments/plans/defense_repeats_20260904/PROTOCOL.md')];
const C = {bg:'#F7F8FA', white:'#FFFFFF', ink:'#17283D', muted:'#516174', blue:'#2468C9', orange:'#D27820', teal:'#147D75', grid:'#DCE2E9', dark:'#14243A', pale:'#B9D6FC'};
const P = Presentation.create({slideSize:{width:1280,height:720}});
P.theme.colorScheme = {name:'Research',themeColors:{accent1:C.blue,accent2:C.orange,accent3:C.teal,accent4:'#A44D51',accent5:'#705DA6',accent6:'#8093A8',bg1:C.white,bg2:C.bg,tx1:C.ink,tx2:C.muted,dk1:'#000000',dk2:C.dark,lt1:C.white,lt2:C.grid,hlink:C.blue,folHlink:'#705DA6'}};
const slides = [];
const speaker = [];
const tableOwners = [];
const chartOwners = [];
const source = rel => `${ROOT}/source/${rel}`;
const qualityAudit = JSON.parse(await fs.readFile(path.join(ROOT, 'quality_metrics_audit.json'), 'utf8'));
const coreMetrics = qualityAudit.groups['Core total'];
const sweMetrics = qualityAudit.groups['SWE Pro'];
const modes = ['single', 'multi-graph', 'multi-orch-guarded'];
const rateText = r => `${Number((100*r.value).toFixed(1))}% (${r.numerator}/${r.denominator})`;
const metricRow = (label, key, group=coreMetrics, selectedModes=modes) => [label, ...selectedModes.map(mode => typeof group[mode][key] === 'number' ? group[mode][key].toFixed(2) : rateText(group[mode][key]))];
function text(s, value, x, y, w, h, size=28, opts={}) {
  const sh=s.shapes.add({geometry:'textbox',name:opts.name??`text-${s.shapes.items?.length??0}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;
  sh.text.style={typeface:FONT,fontSize:size,color:opts.color??C.ink,bold:opts.bold??false,alignment:opts.align??'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0},...opts.style};
  return sh;
}
function newSlide(title, note, sources=[], {dark=false, appendix=false}={}) {
  const s=P.slides.add(); s.background.fill=dark?C.dark:C.bg;
  const n=slides.length+1; slides.push(s);
  if(title) text(s,title,72,50,1125,100,46,{bold:true,color:dark?C.white:C.ink,name:'slide-title'});
  text(s,String(n).padStart(2,'0'),1180,670,40,24,17,{align:'right',color:dark?C.pale:C.muted,name:'slide-number'});
  if(appendix) text(s,'APPENDIX',72,670,150,24,17,{color:C.muted,name:'appendix-label'});
  const notes=`${note}\n\nSources:\n${sources.join('\n')}`;
  s.speakerNotes.textFrame.setText(notes);
  speaker.push({slide:n,title,notes}); return s;
}
function table(s, values, widths, {x=72,y=200,w=1136,h=340,size=25,rowHeight=80,headerHeight=66}={}) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
  t.styleOptions={headerRow:false,bandedRows:false};
  t.borders.assign({fill:C.grid,width:0.7,style:'solid'});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:16,right:12,top:13,bottom:12}});
  for(let r=0;r<values.length;r++) {
    if(t.rows?.[r]) t.rows[r].height=r===0?headerHeight:rowHeight;
    for(let c=0;c<values[r].length;c++) {
      const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2===1?C.white:'#EFF2F6');
      cell.text.style={typeface:FONT,fontSize:r===0?size-1:size,color:r===0?C.white:C.ink,bold:r===0,verticalAlignment:'middle',autoFit:'none',alignment:'left'};
    }
  }
  if(!tableOwners.includes(slides.length)) tableOwners.push(slides.length);return t;
}
function chart(s, {cats,vals,labels,title,x,y,w,h,max=1,format='0%',colors=[C.blue,C.orange,C.teal],horizontal=false,size=23}) {
  const config={position:{left:x,top:y,width:w,height:h},categories:cats,
    series:[{name:title,values:vals.map(v=>Number(v.toFixed(6))),valuesFormatCode:format,points:vals.map((v,i)=>({idx:i,fill:colors[i%colors.length],line:{fill:'none',width:0}})),dataLabelOverrides:labels?.map((v,i)=>({idx:i,text:v,showValue:false,position:'outEnd',textStyle:{typeface:FONT,fontSize:25,bold:true,fill:C.ink}}))}],
    barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:90,varyColors:true},hasLegend:false,
    chartFill:C.bg,chartLine:{fill:'none',width:0},plotAreaFill:C.bg,plotAreaLine:{fill:'none',width:0},
    xAxis:{visible:true,textStyle:{typeface:FONT,fontSize:size,fill:C.muted},line:{fill:C.grid,width:1},majorGridlines:null},
    yAxis:{visible:true,min:0,max,majorUnit:max===1?0.25:undefined,numberFormatCode:format,textStyle:{typeface:FONT,fontSize:20,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.grid,width:0.7}},
    dataLabels:{showValue:!labels,position:'outEnd',textStyle:{typeface:FONT,fontSize:25,bold:true,fill:C.ink}}};
  const ch=s.charts.add('bar',config); applyPresentationChartFont(ch,{fontFamily:FONT});
  if(!chartOwners.includes(slides.length))chartOwners.push(slides.length);return ch;
}

// 01. Cover
{
 const s=newSlide('', 'My project evaluates single-agent and multi-agent LLM architectures for autonomous software development at repository level. The defense primarily covers origin/master at commit 8da3f3f. I will also show a separately labeled extension from kernel/v2. The question is whether specialized roles improve the final solution enough to justify their resource cost. The contribution is an executable evaluation framework and an empirical comparison, rather than training a new language model.', [source('README.md'),source('docs/final_benchmark_results.md')],{dark:true});
 text(s,'Single-Agent and\nMulti-Agent LLMs',72,118,1120,212,78,{bold:true,color:C.white});
 text(s,'Autonomous software development\nat repository level',76,376,1070,100,38,{color:C.pale});
 text(s,'Roman Avanesov\nSV 88/2024',76,558,700,75,27,{color:C.white});
 text(s,'Primary study: master 8da3f3f',795,610,410,32,23,{color:C.pale,align:'right'});
}
// 02. Research question
{
 const s=newSlide('Research question', 'The initial hypothesis was that a single agent would be more cost-effective on simple tasks, while specialized roles would become more useful as complexity increased. Existing research argues that a strong single-agent baseline can match homogeneous multi-agent workflows when computation and context are considered. This project examines that question in an existing repository, where the agent must find files, modify code, run tests, and repair failures. These are hypotheses and motivations, not conclusions established in advance.', ['https://arxiv.org/abs/2601.12307','https://arxiv.org/abs/2604.02460',source('docs/spec.md')]);
 text(s,'Does role specialization improve results\nenough to justify its additional cost?',72,193,1135,160,46,{bold:true});
 text(s,'H1. Simple tasks',72,426,490,42,29,{bold:true,color:C.blue});
 text(s,'Single agents are\nmore cost-effective',72,486,510,110,30);
 text(s,'H2. Increasing complexity',670,426,530,42,29,{bold:true,color:C.orange});
 text(s,'Multi-agent benefits grow with\ntask and codebase complexity',670,486,520,110,30);
}
// 03. Runtime
{
 const s=newSlide('Repository development loop', 'Each run starts from a task description and an initial repository state. The model proposes structured actions. File tools operate within the working repository, while shell commands and tests run in Docker. The agent receives observations and revises its plan or code. After the run, an external evaluator checks the final workspace and records the result. The agent runtime, executor, and evaluator are separate modules. This separation lets the experiment compare workflow policies while reusing much of the execution infrastructure.', [source('src/benchmark/runner.py'),source('src/agents/executor.py'),source('src/benchmark/evaluation.py')]);
 const items=[['01','Task input','Repository snapshot, task description, visible tests'],['02','Agent actions','Read and search files, plan, edit code, run tests'],['03','Revision loop','Use tool feedback until completion or a budget limit'],['04','External evaluation','Check the final patch, required tests, and regressions']];
 items.forEach((it,i)=>{const y=190+i*108;text(s,it[0],72,y,85,53,40,{color:C.blue,bold:true});text(s,it[1],190,y,990,40,30,{bold:true});text(s,it[2],190,y+47,990,50,27,{color:C.muted});});
}
// 04. Dataset
{
 const s=newSlide('Benchmark composition', 'The primary comparison contains 39 tasks and 102 scored runs. The core has 24 tasks from 17 Python repositories, split into 12 small and 12 medium tasks. These are tasks, not 24 distinct repositories. The extension contains 15 selected tasks from an adapted SWE-bench Pro subset. Two selected local-large SQLGlot tasks did not produce a completed paired comparison and are excluded. The task count met the project minimum, but it was not a statistical power calculation.', [source('eval/task_sets/final_v1.json'),source('repositories/collection.csv'),source('docs/final_benchmark_results.md')]);
 text(s,'39',72,164,220,104,84,{bold:true,color:C.blue});text(s,'unique tasks',252,207,325,40,29);
 text(s,'102',686,164,240,104,84,{bold:true,color:C.ink});text(s,'scored runs',934,207,265,40,29);
 table(s,[['Block','Tasks','Repositories','Architectures'],['Core small','12','11','Single, graph, guarded'],['Core medium','12','6','Single, graph, guarded'],['SWE Pro subset','15','3','Single, graph']],[340,160,210,426],{y:308,h:292,size:25,rowHeight:73,headerHeight:65});
 text(s,'Core: 24 tasks from 17 Python repositories. Two unpaired SQLGlot tasks are excluded.',72,625,1110,42,21,{color:C.muted});
}
// 05. Task construction
{
 const s=newSlide('Task construction and validation', 'The core reconstructs real upstream changes, mainly merged pull requests. The starting commit precedes the change. The task description explains the expected behavior without providing the solution diff. Visible tests should pass before the task. Semantic hidden tests should fail before the change and pass with the reference solution. Compatibility checks should pass in both states. Local task selection and descriptions involve manual preparation, while validation and execution are automated. An importer automates the SWE Pro adaptation.', [source('repositories/COLLECT_INSTRUCTION.md'),source('src/benchmark/validate.py'),source('src/benchmark/import_swebench_pro.py')]);
 table(s,[['Test group','Base state','Reference solution','Agent access'],['Visible tests','Pass','Pass','During development'],['Hidden semantic','Fail','Pass','Final evaluation only'],['Compatibility','Pass','Pass','Depends on the suite']],[360,200,250,326],{y:188,h:335,size:25,rowHeight:88,headerHeight:70});
 text(s,'Real upstream changes provide the starting point and reference behavior.',72,553,1120,70,30,{bold:true});
 text(s,'Local descriptions are curated. The SWE Pro importer and validation pipeline automate later steps.',72,618,1120,42,21,{color:C.muted});
}
// 06. Architectures
{
 const s=newSlide('Three architectures', 'The single agent handles the complete cycle and includes guards against prolonged research without progress and a compatibility check before finishing. Multi-graph executes planner, developer, tester, and reviewer roles with deterministic routing. The guarded orchestrator delegates through an LLM supervisor, while code enforces phase constraints and reserves verification steps. The multi-agent roles execute sequentially and share the repository and global budget, with separate role contexts. These are concrete systems with different control rules, not an isolated test of agent count.', [source('src/agents/single_agent.py'),source('src/agents/multi_agent.py'),source('src/agents/roles.py')]);
 table(s,[['Architecture','Control policy','Responsibility'],['Single','One development loop','Research, code, test, repair'],['Multi-graph','Deterministic role routing','Planner, developer, tester, reviewer'],['Guarded','Supervisor with phase rules','Delegation with enforced checks']],[280,400,456],{y:197,h:334,size:25,rowHeight:88,headerHeight:70});
 text(s,'Shared executor and evaluator. Sequential roles with separate contexts.',72,578,1125,85,30,{bold:true});
}
// 07. Metrics
{
 const s=newSlide('Evaluation and agent metrics', 'The main outcome is task_success from the evaluator, not the agent saying it is solved. Success requires visible tests, every required hidden suite, no measured regressions, and no detected test-oracle tampering. Resolved at one summarizes the first final attempt. The system also records tool-use validity, repair behavior, tokens, calls, duration, and cost. Some named AI metrics are operational proxies. For example, patch validity measures whether files changed, and hallucinated references counts malformed or invalid actions. Those limitations are documented explicitly.', [source('src/benchmark/evaluation.py'),source('src/metrics/compute.py'),source('src/metrics/records.py')]);
 text(s,'Final success requires',72,174,1100,48,32,{bold:true});
 text(s,'Visible and required hidden tests pass\nNo regressions or detected test-oracle tampering',72,236,1120,112,35,{bold:true,color:C.blue});
 text(s,'Outcome and efficiency',72,406,515,43,29,{bold:true});
 text(s,'Resolved@1\nCost per solved task\nTokens, calls, steps, duration',72,468,510,140,29,{color:C.muted});
 text(s,'Behavioral diagnostics',678,406,530,43,29,{bold:true});
 text(s,'Tool-use validity\nRepair and visible/final gaps\nExplicitly defined proxy metrics',678,468,525,140,29,{color:C.muted});
}
// 08. Protocol and local pilot
{
 const s=newSlide('Model choice and execution budgets', 'The author reports that preliminary local trials were too slow and produced insufficient solution quality for the selected repository tasks. This was a qualitative feasibility observation, without a quantified pilot table in the saved master artifacts. The primary comparison therefore uses hosted GLM-5.2 through OpenRouter. Core-small uses the same 50-step cap for all architectures. Medium and SWE Pro allow multi-agent systems 75 steps. The SWE Pro initial max_tokens setting is also unequal: 16,384 for single and 4,096 for graph. A truncation retry can increase the initial setting, so these are not absolute per-call limits. Equal steps still do not mean equal token usage or compute. Report the recorded system policies by block and actual resource use.', [source('docs/final_benchmark_results.md'),source('docs/final_benchmark_task_set.md'),source('experiments/results/sessions/final_v1_swepro_single50/sweep.json'),source('experiments/results/sessions/final_v1_swepro_graph75/sweep.json'),'Author-provided explanation of preliminary local trials, September 4, 2026.']);
 text(s,'Primary model: GLM-5.2 through OpenRouter',72,167,1128,58,35,{bold:true,color:C.blue});
 table(s,[['Block','Single','Multi-agent'],['Core small','50 steps','50 steps'],['Core medium','50 steps','75 steps'],['SWE Pro subset','50 steps','75 steps']],[616,260,260],{y:255,h:282,size:25,rowHeight:68,headerHeight:66});
 text(s,'SWE Pro initial output limit: single 16,384 tokens, graph 4,096.',72,545,1125,30,22,{color:C.muted});
 text(s,'Local pilot: slow execution and insufficient solution quality',72,590,1125,38,27,{bold:true});
 text(s,'Author-reported feasibility observation. No quantified pilot results in the saved artifacts.',72,638,1125,28,20,{color:C.muted});
}
// 09. Block results
{
 const s=newSlide('Success rates by benchmark block', 'All three architectures resolve 11 of the 12 small tasks. On medium tasks, single resolves 11 while graph and guarded resolve 10 each. On the selected SWE Pro subset, single and graph both resolve 9 of 15, with identical paired outcomes. Guarded was not run on that extension. The figure presents recorded outcomes, not a statistically established ranking. There is one final attempt per task and architecture. The medium and SWE blocks use different step caps across architectures. SWE Pro also uses different initial completion-token settings (16,384 single and 4,096 graph, before any truncation retry). Core-small is the equal-step comparison.', [source('experiments/results/final_report/summary.csv'),source('docs/final_benchmark_results.md')]);
 ['Core small','Core medium','SWE Pro subset'].forEach((t,i)=>text(s,t,72+i*396,163,362,43,31,{bold:true}));
 chart(s,{cats:['Single','Graph','Guarded'],vals:[11/12,11/12,11/12],labels:['11/12','11/12','11/12'],title:'Core small success rate',x:54,y:221,w:378,h:363,size:20});
 chart(s,{cats:['Single','Graph','Guarded'],vals:[11/12,10/12,10/12],labels:['11/12','10/12','10/12'],title:'Core medium success rate',x:450,y:221,w:378,h:363,size:20});
 chart(s,{cats:['Single','Graph'],vals:[9/15,9/15],labels:['9/15','9/15'],title:'SWE Pro success rate',x:846,y:221,w:378,h:363,size:22});
 text(s,'50 steps for all',72,606,360,33,22,{color:C.muted});text(s,'Single 50 / multi 75',468,606,360,33,22,{color:C.muted});text(s,'Single 50 / graph 75',864,606,350,33,22,{color:C.muted});
}
// 10. Functional quality results
{
 const s=newSlide('Functional quality results', 'These figures reaggregate the 72 primary core runs: 24 tasks for each of the three architectures. Resolved at one is 22 of 24 for single and 21 of 24 for graph and guarded. Visible suites pass in 24, 22, and 23 runs respectively. Required hidden suites pass in 22, 21, and 21 runs. These rates count complete suites per run, not individual test cases. One graph run on sqlparse_pr_746 records three regressed tests, so its regression-run rate is 1 of 24. No handoffs occurred. The aggregate combines equal 50-step caps on small tasks and unequal 50/75-step caps on medium tasks. The rubric quality score is unavailable for all 102 primary final runs. The existing reviewer produces one overall 0-to-5 LLM-judge score from the task and diff, guided by correctness, minimality, maintainability, robustness, and safety. None of the historical review run IDs matches a primary final run. Functional test evidence therefore does not establish superior maintainability, robustness, or safety.', [source('experiments/results/runs.jsonl'),source('src/metrics/compute.py'),source('src/metrics/quality.py'),source('docs/final_benchmark_results.md'),`${ROOT}/quality_metrics_audit.json`]);
 text(s,'Core: 24 tasks per architecture. Small 50/50 steps, medium 50/75.',72,151,1130,43,27,{color:C.muted});
 table(s,[['Metric','Single','Graph','Guarded'],
   metricRow('Resolved@1','resolved_at_1'),
   metricRow('Visible suite pass rate','visible_test_pass_rate'),
   metricRow('Required hidden suite pass','hidden_test_pass_rate'),
   metricRow('Runs with regressions','regression_rate'),
   metricRow('Human handoff rate','handoff_rate')
 ],[436,233,233,234],{y:215,h:371,size:24,rowHeight:62,headerHeight:61});
 text(s,'Rubric quality score: N/A for all 102 final runs.',72,609,1120,35,27,{bold:true,color:C.blue});
 text(s,'No measured ranking for maintainability, robustness, or safety.',72,649,1080,28,22,{color:C.muted});
}
// 11. Measured AI and agent diagnostics
{
 const s=newSlide('AI and agent metric results', 'These figures use the same 24 core tasks per architecture. Tool-use validity pools categorized actions: single 462 of 481, graph 729 of 765, guarded 692 of 719. It measures whether the action is categorized as valid, not whether every tool call achieves its goal. The remaining starred metrics are proxies whose definitions matter. Patch validity means a nonempty changed-files list and is 100 percent for all three systems. Hallucinated references means invalid, malformed, or missing actions per run: 19/24, 32/24, and 25/24, rounded to 0.79, 1.33, and 1.04. It does not count verified nonexistent APIs. Test overfitting is the final-failure fraction among visible-pass runs: 2/24, 1/22, and 2/23. A smaller visible/final gap does not establish better overall performance because denominators differ. Repair success means visible-pass among runs with more than one test iteration: 21/21, 16/17, and 18/19. The metric does not establish that a failure preceded the final pass. Mean test iterations are 3.00, 2.29, and 2.38, and mean LLM calls are 20.71, 37.00, and 40.00. These diagnostics show behavior within this implementation, not a general code-quality ranking.', [source('src/metrics/compute.py'),source('src/metrics/records.py'),`${ROOT}/quality_metrics_audit.json`]);
 text(s,'Core: 24 tasks per architecture. Counts show each denominator.',72,151,1130,43,27,{color:C.muted});
 table(s,[['Metric','Single','Graph','Guarded'],
   metricRow('Patch validity*','patch_validity_rate'),
   metricRow('Tool-use validity','tool_use_validity_rate'),
   metricRow('Hallucinated refs / run*','hallucinated_refs_per_run'),
   metricRow('Test overfitting*','test_overfitting_rate'),
   metricRow('Repair success*','repair_success_rate')
 ],[436,233,233,234],{y:215,h:371,size:23,rowHeight:62,headerHeight:61});
 text(s,'*Operational proxies. Exact definitions appear in the appendix.',72,610,1130,36,26,{bold:true,color:C.blue});
 text(s,'Changed files and valid actions do not establish code quality.',72,650,1100,28,22,{color:C.muted});
}
// 12. Cost efficiency
{
 const s=newSlide('Graph cost more on the 39 shared tasks', 'Across the 39 tasks evaluated by both architectures, single resolves 31 and graph resolves 30. Graph costs 18.6 percent more overall. Its cost per solved task is about 0.1644 dollars, compared with 0.1342 for single. The denominator includes solved tasks, while the numerator includes the cost of both successful and unsuccessful attempts. This pooled comparison is secondary because it combines different step policies and unequal initial completion-token settings in the SWE Pro block. It supports an empirical cost-efficiency finding for this configuration, without establishing universal superiority.', [source('experiments/results/final_report/summary.csv'),`${ROOT}/audit.json`]);
 text(s,'18.6%',72,180,490,136,104,{bold:true,color:C.orange});
 text(s,'higher total cost for graph',76,321,490,90,32,{bold:true});
 table(s,[['Metric','Single','Graph'],['Resolved','31/39','30/39'],['Total cost','$4.1605','$4.9328'],['Cost / solved task','$0.1342','$0.1644']],[332,192,192],{x:520,y:189,w:716,h:365,size:27,rowHeight:92,headerHeight:70});
 text(s,'Secondary aggregate across mixed step policies',72,608,1100,42,25,{color:C.muted});
}
// 13. Paired outcomes
{
 const s=newSlide('Only three tasks changed the ranking', 'The paired comparison is more informative than looking at two percentages in isolation. Both systems solve 29 tasks, and neither solves 7. Single uniquely solves 2, while graph uniquely solves 1. That leaves only three disagreements out of 39. The feature subset favors graph by one task, but it is the same graph-only success rather than an independent confirmation. The experiment does not establish a statistically convincing difference or equivalence. Repeated runs and a larger independent task sample would be needed.', [source('docs/final_benchmark_results.md'),`${ROOT}/audit.json`]);
 table(s,[['Paired outcome','Tasks'],['Both solved','29'],['Neither solved','7'],['Single only','2'],['Graph only','1']],[760,376],{y:190,h:356,size:28,rowHeight:73,headerHeight:65});
 text(s,'A one-task aggregate difference is weak evidence for a general ranking.',72,583,1120,86,32,{bold:true,color:C.blue});
}
// 14. Limitations
{
 const s=newSlide('Limits of the primary experiment', 'The primary experiment uses one model and one final attempt per configuration. Twenty of the 39 shared tasks had exploratory exposure during system development. Most core reference changes are local: 19 of 24 touch one source file, so larger repositories do not necessarily mean harder cross-module changes. Size labels have ambiguities, and the literal repository-count commitment changed to task counts. Dataset files exist locally but need separate packaging for reproduction. The final patches did not receive a new blinded maintainability review.', [source('docs/final_benchmark_results.md'),source('repositories/collection.csv'),`${ROOT}/DEFENSE_EN.md`]);
 const lines=[['One model and one final attempt','Small effects and run-to-run stability remain uncertain'],['Limited complexity coverage','19 of 24 core patches change one source file'],['Development exposure','20 of 39 shared tasks appeared in exploratory runs'],['Rubric quality review and reproduction','No rubric scores for 102 final runs. Dataset packaging remains incomplete.']];
 lines.forEach((it,i)=>{const y=180+i*111;text(s,it[0],72,y,1120,43,31,{bold:true});text(s,it[1],72,y+47,1120,52,26,{color:C.muted});});
}
// 15. Separate extension
{
 const s=newSlide('Kernel v2: initial comparisons', 'Kernel v2 strengthens context accounting, test-evidence validation, execution isolation, and run fingerprints. Its completed core-small blocks show different architecture rankings across configurations. GLM gives 10, 9, and 10 successes. DeepSeek gives 9, 11, and 11. Codex Mini gives 9, 7, and 8. In the original DeepSeek attempt, graph has a slightly lower cost per solved task than single. The following slides report two additional repetitions and show how that relationship changes. However, reasoning and compaction settings differ between model configurations. These observations do not isolate model identity, and they do not establish a complexity effect. The 75-step extension remains separate.', [`${ROOT}/kernel_v2_interim_snapshot.json`,'/Users/tortalpha/Code/repo-level-dev-agent-eval/docs/agent_kernel_v2.md','/Users/tortalpha/Code/repo-level-dev-agent-eval/experiments/results/kernel_v2_report/summary.csv']);
 text(s,'Initial core-small comparisons at a 50-step cap',72,158,1120,50,31,{color:C.muted});
 table(s,[['Model configuration','Single','Graph','Guarded'],['GLM-5.2','10/12','9/12','10/12'],['DeepSeek V4 Flash','9/12','11/12','11/12'],['GPT-5.1 Codex Mini','9/12','7/12','8/12']],[596,180,180,180],{y:238,h:320,size:26,rowHeight:83,headerHeight:65});
 text(s,'Architecture rankings vary across configurations.',72,584,1120,43,30,{bold:true,color:C.blue});
 text(s,'Different reasoning and compaction settings. These results do not extend the master totals.',72,631,1110,31,20,{color:C.muted});
}

// Kernel v2 repeated outcomes
{
 const s=newSlide('Repeated runs: a smaller graph advantage', 'The DeepSeek V4 Flash extension now contains the original comparison plus two additional repetitions. Each attempt uses the same 12 core-small tasks, the same 50-step limit, low reasoning effort, temperature zero, a 48,000-token context budget, and checkpoint compaction. All 48 new observations completed, with no new infrastructure failures. The historical comparison includes the scored tenacity recovery, which used the same recorded setup as the repeats. Graph solves two more tasks in the original attempt, ties single in repeat 1, and solves one more task in repeat 2. Across all 72 observations, single succeeds 29/36 times and graph 32/36. These are repeated outcomes on 12 exposed tasks. They do not provide 36 independent tasks per architecture, a complexity experiment, or a held-out generalization result. Interpret the advantage descriptively.', repeatSources);
 text(s,'DeepSeek V4 Flash. The same 12 tasks and 50-step limit.',72,155,1130,46,29,{color:C.muted});
 const rows=[['Attempt','Single','Graph','Graph lead']];
 for(const [key,label] of [['original','Original'],['r1','Repeat 1'],['r2','Repeat 2']]) {
  const a=RA.repetitions[key].architectures;
  const lead=a['multi-graph'].task_success.numerator-a.single.task_success.numerator;
  rows.push([label,RF(a.single.task_success),RF(a['multi-graph'].task_success),lead===0?'Tie':`+${lead} task${lead===1?'':'s'}`]);
 }
 rows.push(['All attempts',RF(RG.single.task_success),RF(RG['multi-graph'].task_success),'+3 outcomes']);
 table(s,rows,[420,220,220,276],{y:222,h:356,size:28,rowHeight:73,headerHeight:64});
 text(s,`Solved in all three attempts: single ${RA.task_consistency.single.all_three_solved}/12 tasks, graph ${RA.task_consistency['multi-graph'].all_three_solved}/12.`,72,600,1130,40,28,{bold:true,color:C.blue});
 text(s,'Repeated outcomes on 12 familiar tasks. Graph tied single in one repetition.',72,648,1090,28,22,{color:C.muted});
}
// Kernel v2 repeated cost efficiency
{
 const s=newSlide('Single cost less in both new repetitions', 'Every selected observation has complete provider-reported billing. The table divides all billed LLM cost in a block, including failed task attempts, by the number of successful outcomes in that block. It does not include host electricity or hardware cost. Graph has a slightly lower cost per successful outcome in the original comparison, but a higher value in both new repetitions. Across all three attempts, single costs $0.78458366 for 29 successful outcomes, and graph costs $1.04365076 for 32. Thus graph costs about 33.0 percent more overall and 20.5 percent more per successful outcome. The two new repetitions alone cost $1.03556538 in total. Differences in token usage and cache hits contribute to cost variability. Both systems solve the same 11 distinct tasks at least once. The parse_pr_165 task fails in all three attempts for both. This supports a cost versus consistency tradeoff within the tested configurations, not a universal ranking.', repeatSources);
 text(s,'Provider-billed LLM cost per successful outcome',72,156,1130,46,30,{color:C.muted});
 const rows=[['Attempt','Single','Graph']];
 for(const [key,label] of [['original','Original'],['r1','Repeat 1'],['r2','Repeat 2']]) {
  const a=RA.repetitions[key].architectures;
  rows.push([label,USD(a.single.provider_cost_per_success_usd),USD(a['multi-graph'].provider_cost_per_success_usd)]);
 }
 rows.push(['All attempts',USD(RG.single.provider_cost_per_success_usd),USD(RG['multi-graph'].provider_cost_per_success_usd)]);
 table(s,rows,[600,268,268],{y:222,h:356,size:29,rowHeight:73,headerHeight:64});
 text(s,`Total billed: single ${USD(RG.single.effective_cost_usd)}, graph ${USD(RG['multi-graph'].effective_cost_usd)}.`,72,597,1130,43,29,{bold:true,color:C.blue});
 text(s,'Both architectures solved the same 11 distinct tasks at least once.',72,648,1090,28,23,{color:C.muted});
}
// Kernel v2 functional quality and AI diagnostics
{
 const s=newSlide('Repeated-run quality and agent metrics', 'The denominator is 36 scored attempts per architecture on the same 12 tasks, comprising the original comparison and two repetitions. All visible suites pass. Required hidden-suite success is 29/36 for single and 32/36 for graph, and no run records a visible-test regression. The hidden compatibility result reuses the visible suite on every task. Patch presence measures a nonempty changed-files list: 31/36 and 36/36. Tool-use validity pools recognized action categories, including finish and other non-tool events, rather than measuring semantic usefulness or runtime success. Counts are 1002/1033 and 1342/1384. Hallucinated-reference proxies count invalid, malformed, or missing actions: 30/36 and 31/36 events per run. The visible/final gap is 7/36 versus 4/36 and does not prove intentional overfitting. Repair success is visible-pass among attempts with more than one test iteration, 23/23 and 29/29, without requiring a recorded failure-to-pass transition. Single records handoff status in 7/36 attempts and graph in 0/36, which is an agent status rather than evidence that a human intervened. All 72 rubric quality scores remain unavailable. No validated ranking of readability, maintainability, robustness, or safety follows from these diagnostics.', [...repeatSources,path.join(REPO,'src/metrics/compute.py')]);
 text(s,'Original plus two repeats: 36 attempts per architecture, 12 tasks.',72,149,1130,40,27,{color:C.muted});
 const metric=(label,key,percent=true)=>[label,...RM.map(m=>percent?RP(RG[m][key]):RF(RG[m][key]))];
 const t=table(s,[['Metric','Single','Graph'],
  metric('Required hidden suite pass','required_hidden_suite_pass'),
  metric('Visible suite pass','visible_test_pass'),
  metric('Runs with regressions','regression_run_rate'),
  metric('Patch presence*','patch_presence_proxy'),
  metric('Tool-use validity*','tool_use_validity_histogram'),
  ['Hallucinated refs / attempt*',...RM.map(m=>RG[m].hallucinated_reference_proxy_events_per_run.toFixed(2))],
  metric('Repair success*','repair_success_proxy'),
  metric('Visible / final gap*','visible_success_final_failure_proxy')
 ],[558,289,289],{y:203,h:426,size:23,rowHeight:47,headerHeight:50});
 t.cells.block({row:0,column:0,rowCount:9,columnCount:3}).assign({margins:{left:16,right:12,top:7,bottom:7}});
 text(s,'*Operational proxies. Rubric quality scores: N/A for all 72 attempts.',72,651,1105,28,22,{bold:true,color:C.blue});
}

// Original hypotheses and extension conclusions
{
 const s=newSlide('Hypotheses and conclusions', 'The evidence supports a scoped conclusion. H1 predicted that single agents would be more cost-effective on simple tasks. Master supports this descriptively: all systems solve 11/12 core-small tasks, and single has the lowest cost. The DeepSeek kernel/v2 repetitions show mixed cost-efficiency outcomes under the same recorded configuration. Single costs less per successful outcome in both new repeats, but graph is slightly cheaper by that measure in the original attempt. Graph solves 10/12 tasks in all three attempts, compared with 7/12 for single. H2 predicted that the multi-agent benefit would grow with complexity. Master does not support that pattern: single solves 11/12 medium tasks versus 10/12 for both multi-agent variants, while single and graph both solve 9/15 SWE Pro tasks. Unequal step and initial completion-token settings in larger blocks, limited cross-module changes, and the small task sample prevent a general rejection of H2. The new repetitions use small, familiar tasks and do not test H2. The repeatability question was added after the original result, rather than being an initial preregistered hypothesis. Graph leads by 2, 0 and 1 tasks in the three comparisons. Graph is more consistently successful on this sample, while single has lower pooled cost per success. The experiment establishes neither a universal winner nor superior maintainability.', [source('docs/spec.md'),source('docs/final_benchmark_results.md'),...repeatSources],{dark:true});
 text(s,'H1. Single is more cost-effective on simple tasks',72,164,1130,55,33,{bold:true,color:C.white});
 text(s,'Supported in master. Mixed across kernel v2 attempts.',72,223,1130,46,28,{bold:true,color:C.pale});
 text(s,'Single costs less per success in both new DeepSeek repetitions.',72,276,1125,46,27,{color:C.white});
 text(s,'H2. Multi-agent benefits increase with complexity',72,353,1130,55,33,{bold:true,color:C.white});
 text(s,'Not supported in master. The new repeats do not test H2.',72,412,1130,46,28,{bold:true,color:C.pale});
 text(s,'Most core changes are local. Larger blocks use unequal budgets.',72,465,1125,46,27,{color:C.white});
 text(s,'Repeatability finding',72,543,1120,43,31,{bold:true,color:C.pale});
 text(s,'Graph solved 10/12 tasks in all three attempts; single solved 7/12.',72,594,1125,69,28,{color:C.white});
}
// 17. Appendix: dataset questions
{
 const s=newSlide('Dataset definitions and scope changes', 'The original proposal defined small repositories as 500 to 3,000 Python source lines and 5 to 30 source files, and medium repositories as 3,001 to 15,000 lines and 31 to 120 files. The later specification prioritizes LOC when measures disagree. In the core, the original combined thresholds hold for 9 of 12 small tasks and 2 of 12 medium tasks when physical lines are used. boltons and pydash fit the medium range only under nonblank counts, a rule not clearly fixed in the metadata. Multiple tasks share repositories. These are limitations to disclose.', [source('docs/spec.md'),source('repositories/README.md'),source('repositories/collection.csv')],{appendix:true});
 table(s,[['Group','Original source LOC','Original files'],['Small','500–3,000','5–30'],['Medium','3,001–15,000','31–120']],[376,400,360],{y:193,h:240,size:27,rowHeight:82,headerHeight:65});
 text(s,'Final core: 12 tasks per group, from 11 small and 6 medium repositories.',72,477,1115,89,29,{bold:true});
 text(s,'Later rules prioritize LOC. Physical and nonblank counts leave two medium labels ambiguous.',72,584,1110,65,25,{color:C.muted});
}
// 18. Appendix: AI metric proxies
{
 const s=newSlide('Operational definitions of AI metrics', 'These names should be interpreted through the actual implementation. Patch validity is the fraction of runs with changed files. Hallucinated references counts invalid, malformed, and missing actions rather than checking whether APIs exist. Repair success is the visible-pass fraction among runs with more than one test iteration, without requiring a preceding failure. Test overfitting is the final-failure fraction among visible-pass runs. They are useful diagnostics with limited construct validity. Functional test outcomes and cost provide the clearest primary evidence.', [source('src/metrics/compute.py'),source('src/metrics/records.py')],{appendix:true});
 table(s,[['Recorded metric','Implemented proxy'],['Patch validity','Nonempty changed-files list'],['Hallucinated references','Invalid, malformed, or missing actions'],['Repair success','Visible pass with more than one test iteration'],['Test overfitting','Final failure despite visible tests passing']],[436,700],{y:190,h:393,size:25,rowHeight:82,headerHeight:65});
 text(s,'Primary evidence: executable correctness, regressions, and resource cost.',72,617,1110,40,25,{bold:true,color:C.blue});
}
// 19. Appendix: additional AI metrics on the SWE Pro subset
{
 const s=newSlide('SWE Pro: quality and agent metrics', 'This appendix reports the 15 primary SWE Pro tasks evaluated by single and graph. Guarded was not run on this subset. Both systems pass all visible suites, yet only 9 of 15 pass the required hidden suites and resolve the task. Thus each has a visible/final gap of 6 of 15, or 40 percent. Neither records a visible-test regression. Tool-use validity is 390/408 for single and 639/661 for graph. Hallucinated-reference proxies are 17/15 and 18/15 per run. The repair proxy is 14/14 for single and 15/15 for graph. Patch validity is 15/15 and human handoff is 0/15 for both, while rubric quality scores remain unavailable. Mean test iterations are 3.33 and 3.47, and mean LLM calls are 31.53 and 49.20. Single uses a 50-step cap and graph 75. The initial completion-token settings are 16,384 and 4,096 respectively, with possible increases after truncation. This is a descriptive comparison under the recorded policies.', [source('src/metrics/compute.py'),`${ROOT}/quality_metrics_audit.json`],{appendix:true});
 text(s,'15 tasks per architecture. Single: 50 steps. Graph: 75 steps.',72,151,1130,40,27,{color:C.muted});
 const sweTable=table(s,[['Metric','Single','Graph'],
   metricRow('Resolved@1 / required hidden pass','resolved_at_1',sweMetrics,modes.slice(0,2)),
   metricRow('Visible suite pass rate','visible_test_pass_rate',sweMetrics,modes.slice(0,2)),
   metricRow('Runs with regressions','regression_rate',sweMetrics,modes.slice(0,2)),
   metricRow('Tool-use validity','tool_use_validity_rate',sweMetrics,modes.slice(0,2)),
   metricRow('Hallucinated refs / run*','hallucinated_refs_per_run',sweMetrics,modes.slice(0,2)),
   metricRow('Test overfitting*','test_overfitting_rate',sweMetrics,modes.slice(0,2)),
   metricRow('Repair success*','repair_success_rate',sweMetrics,modes.slice(0,2))
 ],[560,288,288],{y:211,h:409,size:23,rowHeight:50,headerHeight:59});
 sweTable.cells.block({row:0,column:0,rowCount:8,columnCount:3}).assign({margins:{left:16,right:12,top:8,bottom:8}});
 text(s,'*Same proxy definitions as core. Visible passes do not guarantee final success.',72,633,1130,30,23,{color:C.blue,bold:true});
}
// 20. Appendix: literature
{
 const s=newSlide('Related work and reproducibility', 'Xu and colleagues motivate a stronger single-agent baseline for homogeneous workflows. Tran and Kiela show why compute and context accounting matter in multi-hop reasoning, which is a different domain. Agentless demonstrates the value of simple software-engineering pipelines. The local evidence consists of the frozen final-v1 manifest, run records, patches, and the report generator at the defended commit. Full reproduction additionally requires the task descriptions, hidden tests, and repository snapshots that are stored outside Git.', ['https://arxiv.org/abs/2601.12307','https://arxiv.org/abs/2604.02460','https://arxiv.org/abs/2407.01489',source('eval/task_sets/final_v1.json')],{appendix:true});
 text(s,'Xu et al. (2026)',72,179,1120,40,30,{bold:true});text(s,'Rethinking the Value of Multi-Agent Workflow\nA Strong Single Agent Baseline',72,224,1120,81,27,{color:C.muted});
 text(s,'Tran and Kiela (2026)',72,333,1120,40,30,{bold:true});text(s,'Single-Agent LLMs Outperform Multi-Agent Systems on\nMulti-Hop Reasoning Under Equal Thinking Token Budgets',72,378,1120,85,27,{color:C.muted});
 text(s,'Xia et al. (2024)',72,493,1120,40,30,{bold:true});text(s,'Agentless: Demystifying LLM-based Software Engineering Agents',72,538,1120,75,27,{color:C.muted});
}


// Appendix: task-level repeatability
{
 const s=newSlide('Task outcomes across three attempts', 'Each cell lists the original scored attempt, repeat 1, and repeat 2 in that order. P means final task_success is true and F means it is false. The original comparison includes the successful tenacity recovery session. Single has seven tasks solved in all three attempts, four with mixed results, and one with three failures. Graph has ten tasks solved in all three, one mixed result, and one with three failures. Both solve the same 11 tasks at least once. Across 36 paired task-by-attempt observations, both succeed in 28, both fail in 3, graph alone succeeds in 4, and single alone succeeds in 1. Repetitions within a task are dependent observations, so this table does not establish a statistically general ranking.', repeatSources,{appendix:true});
 text(s,'Order within each cell: original, repeat 1, repeat 2',72,147,1130,38,26,{color:C.muted});
 const taskLabels={h11_pr_181:'h11 #181',humanize_pr_329:'humanize #329',pluggy_pr_646:'pluggy #646',w3lib_pr_272:'w3lib #272',parse_pr_165:'parse #165',parse_pr_227:'parse #227',cachetools_pr_57d2e48:'cachetools 57d2e48',tinydb_pr_616:'tinydb #616',python_dotenv_pr_640:'python-dotenv #640',tenacity_pr_628:'tenacity #628',freezegun_pr_546:'freezegun #546',croniter_pr_235:'croniter #235'};
 const rows=[['Task','Single','Graph'],...RA.task_outcomes.map(r=>[taskLabels[r.task_id],...RM.map(m=>r[m].outcomes_original_r1_r2.map(x=>x?'P':'F').join('/'))])];
 for (let half=0;half<2;half++) {
  const values=[rows[0],...rows.slice(1+half*6,7+half*6)];
  const t=table(s,values,[280,134,134],{x:72+half*588,y:218,w:548,h:350,size:23,rowHeight:50,headerHeight:50});
  t.cells.block({row:0,column:0,rowCount:7,columnCount:3}).assign({margins:{left:16,right:12,top:7,bottom:7}});
 }
 text(s,'P = required evaluation passed. F = final task failure.',72,612,1110,32,24,{color:C.muted});
}

await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(path.join(WORK,'output'),{recursive:true});
await fs.writeFile(path.join(BUILD,'speaker_notes.json'),JSON.stringify(speaker,null,2));
await fs.writeFile(path.join(BUILD,'deck_content.json'),JSON.stringify({font:FONT,slides:slides.length,tableOwners,chartOwners},null,2));
const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(P)).save(candidate);
console.log(JSON.stringify({candidate,slides:slides.length,font:FONT}));
for(let i=0;i<slides.length;i++) {
 const png=await P.export({slide:slides[i],format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await slides[i].export({format:'layout'});
 await fs.writeFile(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.layout.json`),await layout.text());
 console.log(`Rendered ${i+1}/${slides.length}`);
}
const version=process.env.DECK_REVISION??'v8';
const finalPath=path.join(WORK,'output',`Repository_Level_Agent_Evaluation_Defense_EN_${version}.pptx`);
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath,pythonExecutable:PYTHON,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,
 fontPolicy:{basis:'reference',families:[FONT],referencePath:REFERENCE,referenceSha256:REFERENCE_SHA},materializeLiteralChartWorkbooks:true,
 verifyArtifactToolImport:true,receiptPath:path.join(BUILD,`${version}.validation.json`)});
await fs.writeFile(path.join(BUILD,`${version}.finalize.json`),JSON.stringify(result,null,2));
await fs.writeFile(path.join(WORK,'output',`Speaker_Notes_EN_${version}.md`),speaker.map(it=>`**Slide ${it.slide}: ${it.title||'Title'}**\n\n${it.notes}\n`).join('\n'));
console.log(JSON.stringify({finalPath,validation:result}));
