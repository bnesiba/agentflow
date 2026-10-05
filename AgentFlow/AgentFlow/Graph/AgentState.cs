using GraphFlow.flow;
using LLMAbstraction.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AgentFlow.Graph
{
    public class AgentState : IDefaultValueHaver<AgentState>
    {
        public List<UnifiedMessage> Messages { get; set; }
        public static AgentState DefaultValue()
        {
            return new AgentState
            {
                Messages = new List<UnifiedMessage>()
            };
        }
    }
}
