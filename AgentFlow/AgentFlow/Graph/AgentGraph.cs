using AgentFlow.Graph.Edges;
using AgentFlow.Graph.Nodes;
using graphFlow.models;
using graphFlow.util;

namespace AgentFlow.Graph
{
    public class AgentGraph
    {
       private GraphBuilder<AgentState> _graphBuilder;

        public AgentGraph(GraphBuilder<AgentState> graphBuilder)
        {
            _graphBuilder = graphBuilder;
        }

        //Define Graph
        public ExecutableGraph<AgentState> GetGraph()
        {
            ExecutableGraph<AgentState> graph = _graphBuilder.GetExecutableGraph();
            graph.AddNode(PredefinedNodes.LLMNodeId, PredefinedNodes.LLMNode);
            graph.AddNode(PredefinedNodes.ToolNodeId, PredefinedNodes.ToolNode);

            graph.AddEdge(PredefinedNodes.LLMNodeId, PredefinedNodes.ToolNodeId, PredefinedEdges.HasToolCalls);
            graph.AddEdge(PredefinedNodes.ToolNodeId, PredefinedNodes.LLMNodeId);

            graph.SetStartNode(PredefinedNodes.LLMNodeId);
            return graph;
        }
    }
}
