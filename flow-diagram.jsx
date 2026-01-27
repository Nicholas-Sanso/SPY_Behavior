import React from 'react';

export default function DataPipelineFlow() {
  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-slate-800 mb-8">Financial Data Processing Pipeline</h1>
        
        <svg width="1400" height="2400" className="mx-auto">
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
              <polygon points="0 0, 10 3, 0 6" fill="#475569" />
            </marker>
            <filter id="shadow">
              <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.15"/>
            </filter>
          </defs>

          {/* Start: lists of strings */}
          <rect x="50" y="20" width="200" height="60" rx="8" fill="#3b82f6" stroke="#2563eb" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="55" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">lists of strings</text>

          {/* Arrow 1: fetch_statement */}
          <line x1="150" y1="80" x2="150" y2="140" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="260" y="110" fill="#0f172a" fontSize="13" fontWeight="500">fetch_statement</text>

          {/* income_data_2_years */}
          <rect x="50" y="140" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="175" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_data_2_years</text>

          {/* Fork to EDA and main path */}
          <line x1="250" y1="170" x2="350" y2="170" stroke="#475569" strokeWidth="2"/>
          <line x1="350" y1="170" x2="350" y2="240" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="360" y="205" fill="#0f172a" fontSize="13" fontWeight="500">count_problematic_entries</text>
          
          {/* EDA node 1 */}
          <ellipse cx="350" cy="270" rx="40" ry="30" fill="#ef4444" stroke="#dc2626" strokeWidth="2" filter="url(#shadow)"/>
          <text x="350" y="278" textAnchor="middle" fill="white" fontSize="13" fontWeight="600">EDA</text>

          {/* Main path continues */}
          <line x1="150" y1="200" x2="150" y2="340" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="260" y="270" fill="#0f172a" fontSize="13" fontWeight="500">sort_by_symbol_then_date</text>

          {/* income_sorted */}
          <rect x="50" y="340" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="375" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_sorted</text>

          {/* Arrow: compute_log_change */}
          <line x1="150" y1="400" x2="150" y2="460" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="260" y="430" fill="#0f172a" fontSize="13" fontWeight="500">compute_log_change</text>

          {/* income_log_change */}
          <rect x="50" y="460" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="495" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_log_change</text>

          {/* Fork to EDA */}
          <line x1="250" y1="490" x2="350" y2="490" stroke="#475569" strokeWidth="2"/>
          <line x1="350" y1="490" x2="350" y2="540" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="360" y="515" fill="#0f172a" fontSize="13" fontWeight="500">count_zeros_nans_logchg</text>
          
          {/* EDA node 2 */}
          <ellipse cx="350" cy="570" rx="40" ry="30" fill="#ef4444" stroke="#dc2626" strokeWidth="2" filter="url(#shadow)"/>
          <text x="350" y="578" textAnchor="middle" fill="white" fontSize="13" fontWeight="600">EDA</text>

          {/* Main path continues */}
          <line x1="150" y1="520" x2="150" y2="640" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="260" y="580" fill="#0f172a" fontSize="13" fontWeight="500">add_symbol_date</text>

          {/* income_symbol_date_added */}
          <rect x="20" y="640" width="260" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="675" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_symbol_date_added</text>

          {/* Arrow: filter_by_common_pairs */}
          <line x1="150" y1="700" x2="150" y2="760" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="270" y="730" fill="#0f172a" fontSize="13" fontWeight="500">filter_by_common_pairs</text>

          {/* income_post_nans_overlapped */}
          <rect x="10" y="760" width="280" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="795" textAnchor="middle" fill="white" fontSize="13" fontWeight="600">income_post_nans_overlapped</text>

          {/* Arrow: sort_all_financials */}
          <line x1="150" y1="820" x2="150" y2="880" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="270" y="850" fill="#0f172a" fontSize="13" fontWeight="500">sort_all_financials</text>

          {/* income_post_nans_overlapped_sorted */}
          <rect x="0" y="880" width="300" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="915" textAnchor="middle" fill="white" fontSize="13" fontWeight="600">income_post_nans_overlapped_sorted</text>

          {/* Fork to two EDAs */}
          <line x1="300" y1="910" x2="400" y2="910" stroke="#475569" strokeWidth="2"/>
          <line x1="400" y1="910" x2="400" y2="960" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="410" y="935" fill="#0f172a" fontSize="13" fontWeight="500">check_alignment</text>
          
          {/* EDA node 3 */}
          <ellipse cx="400" cy="990" rx="40" ry="30" fill="#ef4444" stroke="#dc2626" strokeWidth="2" filter="url(#shadow)"/>
          <text x="400" y="998" textAnchor="middle" fill="white" fontSize="13" fontWeight="600">EDA</text>

          {/* Another fork to EDA */}
          <line x1="400" y1="910" x2="520" y2="910" stroke="#475569" strokeWidth="2"/>
          <line x1="520" y1="910" x2="520" y2="960" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="530" y="935" fill="#0f172a" fontSize="13" fontWeight="500">outlier_check_1</text>
          
          {/* EDA node 4 */}
          <ellipse cx="520" cy="990" rx="40" ry="30" fill="#ef4444" stroke="#dc2626" strokeWidth="2" filter="url(#shadow)"/>
          <text x="520" y="998" textAnchor="middle" fill="white" fontSize="13" fontWeight="600">EDA</text>

          {/* Main path continues */}
          <line x1="150" y1="940" x2="150" y2="1060" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="280" y="1000" fill="#0f172a" fontSize="13" fontWeight="500">drop_high_zero_columns</text>

          {/* income_after_feature_drop */}
          <rect x="20" y="1060" width="260" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="1095" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_after_feature_drop</text>

          {/* Fork to EDA */}
          <line x1="280" y1="1090" x2="400" y2="1090" stroke="#475569" strokeWidth="2"/>
          <line x1="400" y1="1090" x2="400" y2="1140" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="410" y="1115" fill="#0f172a" fontSize="13" fontWeight="500">outlier_check_1</text>
          
          {/* EDA node 5 */}
          <ellipse cx="400" cy="1170" rx="40" ry="30" fill="#ef4444" stroke="#dc2626" strokeWidth="2" filter="url(#shadow)"/>
          <text x="400" y="1178" textAnchor="middle" fill="white" fontSize="13" fontWeight="600">EDA</text>

          {/* Main path continues */}
          <line x1="150" y1="1120" x2="150" y2="1240" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="280" y="1180" fill="#0f172a" fontSize="13" fontWeight="500">drop_outlier_rows</text>

          {/* income_after_outlier_drop */}
          <rect x="20" y="1240" width="260" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="1275" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_after_outlier_drop</text>

          {/* Arrow: align_dfs_on_symbol_date */}
          <line x1="150" y1="1300" x2="150" y2="1360" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="280" y="1330" fill="#0f172a" fontSize="13" fontWeight="500">align_dfs_on_symbol_date</text>

          {/* income_aligned */}
          <rect x="50" y="1360" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="1395" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_aligned</text>

          {/* Arrow: standardize_as_a_check */}
          <line x1="150" y1="1420" x2="150" y2="1480" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="280" y="1450" fill="#0f172a" fontSize="13" fontWeight="500">standardize_as_a_check</text>

          {/* income_after_outlier_drop_standardized */}
          <rect x="0" y="1480" width="300" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="150" y="1515" textAnchor="middle" fill="white" fontSize="13" fontWeight="600">income_after_outlier_drop_standardized</text>

          {/* Split into two paths */}
          <line x1="150" y1="1540" x2="150" y2="1600" stroke="#475569" strokeWidth="2"/>
          
          {/* Left path: pickle */}
          <line x1="150" y1="1600" x2="80" y2="1600" stroke="#475569" strokeWidth="2"/>
          <line x1="80" y1="1600" x2="80" y2="1660" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="10" y="1630" fill="#0f172a" fontSize="13" fontWeight="500">pickle</text>
          
          {/* .pkl file */}
          <rect x="0" y="1660" width="160" height="60" rx="8" fill="#8b5cf6" stroke="#7c3aed" strokeWidth="2" filter="url(#shadow)"/>
          <text x="80" y="1688" textAnchor="middle" fill="white" fontSize="11" fontWeight="600">income_after_outlier</text>
          <text x="80" y="1704" textAnchor="middle" fill="white" fontSize="11" fontWeight="600">_drop_standardized.pkl</text>

          {/* Right path: run_statement_univariate */}
          <line x1="150" y1="1600" x2="250" y2="1600" stroke="#475569" strokeWidth="2"/>
          <line x1="250" y1="1600" x2="250" y2="1780" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="260" y="1690" fill="#0f172a" fontSize="13" fontWeight="500">run_statement_univariate</text>

          {/* merged_income */}
          <rect x="180" y="1780" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="280" y="1815" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">merged_income</text>

          {/* Arrow: select_significant_features */}
          <line x1="280" y1="1840" x2="280" y2="1900" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="410" y="1870" fill="#0f172a" fontSize="13" fontWeight="500">select_significant_features</text>

          {/* selected_income */}
          <rect x="180" y="1900" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="280" y="1935" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">selected_income</text>

          {/* Split into two paths from selected_income */}
          <line x1="280" y1="1960" x2="280" y2="2020" stroke="#475569" strokeWidth="2"/>
          
          {/* Left path: combined_selected_features - represents selected_balance and selected_cashflow coming in */}
          <line x1="100" y1="2020" x2="280" y2="2020" stroke="#475569" strokeWidth="2"/>
          <line x1="280" y1="2020" x2="280" y2="2100" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="50" y="2010" fill="#0f172a" fontSize="13" fontWeight="600">selected_balance & selected_cashflow</text>
          <text x="310" y="2060" fill="#0f172a" fontSize="13" fontWeight="500">combined_selected_features</text>

          {/* combined_selected_line_items */}
          <rect x="160" y="2100" width="240" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="280" y="2135" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">combined_selected_line_items</text>

          {/* Arrow to regression */}
          <line x1="280" y1="2160" x2="280" y2="2220" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="420" y="2190" fill="#0f172a" fontSize="13" fontWeight="500">regress_log_pe_on_pca</text>

          {/* regression output 1 */}
          <rect x="180" y="2220" width="200" height="60" rx="8" fill="#f59e0b" stroke="#d97706" strokeWidth="2" filter="url(#shadow)"/>
          <text x="280" y="2255" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">regression output</text>

          {/* Right path from selected_income: run_pca */}
          <line x1="380" y1="1930" x2="580" y2="1930" stroke="#475569" strokeWidth="2"/>
          <line x1="580" y1="1930" x2="580" y2="1990" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="590" y="1960" fill="#0f172a" fontSize="13" fontWeight="500">run_pca</text>

          {/* income_df_pca */}
          <rect x="480" y="1990" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="580" y="2025" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_df_pca</text>

          {/* Combined PCA path - represents balance_df_pca and cashflow_df coming in */}
          <line x1="900" y1="2020" x2="580" y2="2020" stroke="#475569" strokeWidth="2"/>
          <line x1="580" y1="2050" x2="580" y2="2100" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="740" y="2010" fill="#0f172a" fontSize="13" fontWeight="600">balance_df_pca & cashflow_df</text>
          <text x="690" y="2075" fill="#0f172a" fontSize="13" fontWeight="500">combined_pca_dfs</text>

          {/* combined_pca_df */}
          <rect x="480" y="2100" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="580" y="2135" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">combined_pca_df</text>

          {/* Arrow to regression */}
          <line x1="580" y1="2160" x2="580" y2="2220" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="690" y="2190" fill="#0f172a" fontSize="13" fontWeight="500">regress_log_pe_on_pca</text>

          {/* regression output 2 */}
          <rect x="480" y="2220" width="200" height="60" rx="8" fill="#f59e0b" stroke="#d97706" strokeWidth="2" filter="url(#shadow)"/>
          <text x="580" y="2255" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">regression output</text>

          {/* Path from .pkl file */}
          <line x1="80" y1="1720" x2="80" y2="1780" stroke="#475569" strokeWidth="2"/>
          <line x1="80" y1="1780" x2="900" y2="1780" stroke="#475569" strokeWidth="2"/>
          <line x1="900" y1="1780" x2="900" y2="2100" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
          <text x="910" y="1940" fill="#0f172a" fontSize="13" fontWeight="500">compute_latent_factors</text>

          {/* income_latent */}
          <rect x="800" y="2100" width="200" height="60" rx="8" fill="#10b981" stroke="#059669" strokeWidth="2" filter="url(#shadow)"/>
          <text x="900" y="2135" textAnchor="middle" fill="white" fontSize="14" fontWeight="600">income_latent</text>

          {/* Legend */}
          <g transform="translate(1050, 50)">
            <text x="0" y="0" fill="#0f172a" fontSize="16" fontWeight="700">Legend</text>
            <rect x="0" y="15" width="60" height="30" rx="4" fill="#3b82f6" stroke="#2563eb" strokeWidth="1"/>
            <text x="70" y="35" fill="#0f172a" fontSize="12">Input Data</text>
            
            <rect x="0" y="55" width="60" height="30" rx="4" fill="#10b981" stroke="#059669" strokeWidth="1"/>
            <text x="70" y="75" fill="#0f172a" fontSize="12">Dataframe</text>
            
            <rect x="0" y="95" width="60" height="30" rx="4" fill="#8b5cf6" stroke="#7c3aed" strokeWidth="1"/>
            <text x="70" y="115" fill="#0f172a" fontSize="12">Pickle File</text>
            
            <ellipse cx="30" cy="145" rx="25" ry="20" fill="#ef4444" stroke="#dc2626" strokeWidth="1"/>
            <text x="70" y="150" fill="#0f172a" fontSize="12">EDA (Dead End)</text>
            
            <rect x="0" y="170" width="60" height="30" rx="4" fill="#f59e0b" stroke="#d97706" strokeWidth="1"/>
            <text x="70" y="190" fill="#0f172a" fontSize="12">Final Output</text>
            
            <line x1="0" y1="220" x2="60" y2="220" stroke="#475569" strokeWidth="2" markerEnd="url(#arrowhead)"/>
            <text x="70" y="225" fill="#0f172a" fontSize="12">Function (on arrow)</text>
          </g>
        </svg>
      </div>
    </div>
  );
}
