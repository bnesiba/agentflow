using LLMAbstraction.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AgentFlow.Graph.Edges
{
    public class PredefinedEdges
    {
        public static bool HasToolCalls(AgentState state)
        {
            return state.Messages.LastMessageIsAssistantWithToolCalls();
        }
    }
}
