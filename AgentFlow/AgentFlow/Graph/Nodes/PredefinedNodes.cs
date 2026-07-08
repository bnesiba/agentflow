using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AgentFlow.Graph.Nodes
{
    public class PredefinedNodes
    {
        public static string LLMNodeId = "LLM";
        public static string ToolNodeId = "Tool";

        public static AgentState LLMNode(AgentState state)
        {
            return state;
        }

        public static AgentState ToolNode(AgentState state)
        {
            return state;
        }
    }
}
