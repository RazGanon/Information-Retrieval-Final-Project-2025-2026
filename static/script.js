// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const clearBtn = document.getElementById('clearBtn');
const searchTypeRadios = document.querySelectorAll('input[name="searchType"]');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const resultsContainer = document.getElementById('resultsContainer');
const resultsList = document.getElementById('resultsList');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
const searchTime = document.getElementById('searchTime');
const searchTypeBadge = document.getElementById('searchTypeBadge');
const emptyState = document.getElementById('emptyState');
const searchHistory = document.getElementById('searchHistory');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

// Search type labels and badges
const searchTypeLabels = {
    'search': 'Main Search (Ensemble)',
    'search_body': 'Body Search',
    'search_title': 'Title Search',
    'search_anchor': 'Anchor Search',
    'search_pagerank': 'PageRank Search',
    'search_pageview': 'PageView Search'
};

const searchTypeBadges = {
    'search': 'ENSEMBLE',
    'search_body': 'BODY',
    'search_title': 'TITLE',
    'search_anchor': 'ANCHOR',
    'search_pagerank': 'PAGERANK',
    'search_pageview': 'PAGEVIEW'
};

// Search history management
let searchHistoryList = JSON.parse(localStorage.getItem('ramewiki_history') || '[]');
const MAX_HISTORY = 10;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateClearButton();
    loadSearchHistory();
    setupKeyboardShortcuts();
    setupExampleQueries();
});

// Update clear button visibility
function updateClearButton() {
    clearBtn.style.display = searchInput.value.trim() ? 'flex' : 'none';
}

// Clear button functionality
clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    searchInput.value = '';
    searchInput.focus();
    updateClearButton();
    hideError();
});

// Search input changes
searchInput.addEventListener('input', updateClearButton);
searchInput.addEventListener('focus', () => {
    if (searchHistoryList.length > 0) {
        searchHistory.style.display = 'block';
    }
});

// Click outside to hide history
document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !searchHistory.contains(e.target)) {
        searchHistory.style.display = 'none';
    }
});

// Handle Enter key press
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// Handle search button click
searchBtn.addEventListener('click', performSearch);

// Keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Focus search with /
        if (e.key === '/' && !e.ctrlKey && !e.metaKey && e.target.tagName !== 'INPUT') {
            e.preventDefault();
            searchInput.focus();
        }
        
        // Clear with Esc
        if (e.key === 'Escape' && document.activeElement === searchInput) {
            searchInput.value = '';
            updateClearButton();
            searchInput.blur();
        }
    });
}

// Example queries
function setupExampleQueries() {
    const exampleTags = document.querySelectorAll('.example-tag');
    exampleTags.forEach(tag => {
        tag.addEventListener('click', () => {
            searchInput.value = tag.textContent;
            updateClearButton();
            performSearch();
        });
    });
}

// Search history functions
function addToHistory(query) {
    // Remove if already exists
    searchHistoryList = searchHistoryList.filter(q => q.toLowerCase() !== query.toLowerCase());
    // Add to beginning
    searchHistoryList.unshift(query);
    // Keep only last MAX_HISTORY
    searchHistoryList = searchHistoryList.slice(0, MAX_HISTORY);
    // Save to localStorage
    localStorage.setItem('ramewiki_history', JSON.stringify(searchHistoryList));
    loadSearchHistory();
}

function loadSearchHistory() {
    if (searchHistoryList.length === 0) {
        searchHistory.style.display = 'none';
        return;
    }
    
    historyList.innerHTML = '';
    searchHistoryList.forEach(query => {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.textContent = query;
        item.addEventListener('click', () => {
            searchInput.value = query;
            updateClearButton();
            performSearch();
        });
        historyList.appendChild(item);
    });
}

clearHistoryBtn.addEventListener('click', () => {
    searchHistoryList = [];
    localStorage.removeItem('ramewiki_history');
    loadSearchHistory();
});

// Perform search
let searchStartTime = 0;

async function performSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showError('Please enter a search query');
        return;
    }

    // Add to history
    addToHistory(query);

    // Get selected search type
    const selectedType = document.querySelector('input[name="searchType"]:checked').value;
    
    // Hide previous results and errors
    hideError();
    hideResults();
    hideEmptyState();
    showLoading();
    searchHistory.style.display = 'none';
    
    // Start timing
    searchStartTime = performance.now();

    try {
        // Make API request
        const url = `/${selectedType}?query=${encodeURIComponent(query)}`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const results = await response.json();
        
        // Calculate search time
        const searchTimeMs = performance.now() - searchStartTime;
        
        hideLoading();
        
        if (results && results.length > 0) {
            // Fetch additional metadata for results
            const enrichedResults = await enrichResults(results);
            displayResults(enrichedResults, query, selectedType, searchTimeMs);
        } else {
            showEmptyState('No results found. Try a different query.');
        }
    } catch (error) {
        hideLoading();
        showError(`Error performing search: ${error.message}`);
        console.error('Search error:', error);
    }
}

// Enrich results with PageRank and PageViews
async function enrichResults(results) {
    if (results.length === 0) return results;

    // Extract document IDs (first element of each tuple)
    const docIds = results.map(result => parseInt(result[0])).filter(id => !isNaN(id));

    if (docIds.length === 0) return results;

    try {
        // Fetch PageRank and PageViews in parallel
        const [pagerankResponse, pageviewResponse] = await Promise.all([
            fetch('/get_pagerank', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(docIds)
            }),
            fetch('/get_pageview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(docIds)
            })
        ]);

        const pageranks = await pagerankResponse.json();
        const pageviews = await pageviewResponse.json();

        // Enrich results with metadata
        return results.map((result, index) => {
            const docId = parseInt(result[0]);
            const docIndex = docIds.indexOf(docId);
            
            return {
                id: result[0],
                title: result[1],
                pagerank: docIndex !== -1 ? pageranks[docIndex] : 0,
                pageviews: docIndex !== -1 ? pageviews[docIndex] : 0
            };
        });
    } catch (error) {
        console.error('Error enriching results:', error);
        // Return results without enrichment if metadata fetch fails
        return results.map(result => ({
            id: result[0],
            title: result[1],
            pagerank: 0,
            pageviews: 0
        }));
    }
}

// Display search results
function displayResults(results, query, searchType, searchTimeMs) {
    resultsTitle.textContent = `Results for "${query}"`;
    resultsCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
    searchTime.textContent = `Found in ${(searchTimeMs / 1000).toFixed(2)}s`;
    searchTypeBadge.textContent = searchTypeBadges[searchType] || 'SEARCH';
    searchTypeBadge.style.display = 'inline-block';
    
    resultsList.innerHTML = '';
    
    results.forEach((result, index) => {
        const resultItem = createResultItem(result, index + 1, query);
        resultsList.appendChild(resultItem);
    });
    
    showResults();
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// Create a result item element
function createResultItem(result, rank, query) {
    const item = document.createElement('div');
    item.className = 'result-item';
    
    const title = result.title || `Document ${result.id}`;
    const pagerank = result.pagerank || 0;
    const pageviews = result.pageviews || 0;
    
    // Format numbers
    const formattedPageRank = pagerank > 0 ? pagerank.toFixed(6) : 'N/A';
    const formattedPageViews = pageviews > 0 ? pageviews.toLocaleString() : 'N/A';
    
    // Highlight query terms in title
    const highlightedTitle = highlightText(title, query);
    
    // Generate snippet (simulated - in real app would come from backend)
    const snippet = generateSnippet(title, query);
    const highlightedSnippet = highlightText(snippet, query);
    
    // Wikipedia URL
    const wikiUrl = `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/\s+/g, '_'))}`;
    
    item.innerHTML = `
        <div class="result-item-header">
            <div>
                <div class="result-title">${highlightedTitle}</div>
            </div>
            <div class="result-id">ID: ${result.id}</div>
        </div>
        ${snippet ? `<div class="result-snippet">${highlightedSnippet}</div>` : ''}
        <div class="result-meta">
            <div class="meta-item">
                <strong>Rank:</strong> ${rank}
            </div>
            <div class="meta-item">
                <strong>PageRank:</strong> ${formattedPageRank}
            </div>
            <div class="meta-item">
                <strong>PageViews:</strong> ${formattedPageViews}
            </div>
        </div>
        <div class="result-actions">
            <button class="copy-link-btn" data-url="${wikiUrl}">
                📋 Copy Link
            </button>
        </div>
    `;
    
    // Add click handler to open Wikipedia article (but not on buttons)
    item.addEventListener('click', (e) => {
        if (!e.target.closest('.copy-link-btn') && !e.target.closest('.result-actions')) {
            window.open(wikiUrl, '_blank');
        }
    });
    
    // Copy link functionality
    const copyBtn = item.querySelector('.copy-link-btn');
    copyBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
            await navigator.clipboard.writeText(wikiUrl);
            copyBtn.textContent = '✓ Copied!';
            copyBtn.classList.add('copied');
            setTimeout(() => {
                copyBtn.textContent = '📋 Copy Link';
                copyBtn.classList.remove('copied');
            }, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = wikiUrl;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            copyBtn.textContent = '✓ Copied!';
            setTimeout(() => {
                copyBtn.textContent = '📋 Copy Link';
            }, 2000);
        }
    });
    
    return item;
}

// Highlight query terms in text
function highlightText(text, query) {
    if (!query || !text) return escapeHtml(text);
    
    const queryTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2);
    let highlighted = escapeHtml(text);
    
    queryTerms.forEach(term => {
        const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
        highlighted = highlighted.replace(regex, '<span class="highlight">$1</span>');
    });
    
    return highlighted;
}

// Generate snippet (simulated)
function generateSnippet(title, query) {
    // In a real implementation, this would come from the backend
    // For now, create a simple snippet based on the title
    const snippets = [
        `Learn more about ${title} on Wikipedia.`,
        `${title} is a topic covered in Wikipedia.`,
        `Find comprehensive information about ${title}.`,
        `Explore ${title} and related topics.`
    ];
    return snippets[Math.floor(Math.random() * snippets.length)];
}

// Utility functions
function showLoading() {
    loadingIndicator.style.display = 'block';
    searchBtn.disabled = true;
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
    searchBtn.disabled = false;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

function showResults() {
    resultsContainer.style.display = 'block';
    emptyState.style.display = 'none';
}

function hideResults() {
    resultsContainer.style.display = 'none';
}

function showEmptyState(message) {
    if (typeof message === 'string') {
        const emptyTitle = emptyState.querySelector('.empty-title');
        const emptySubtitle = emptyState.querySelector('.empty-subtitle');
        if (emptyTitle && emptySubtitle) {
            emptySubtitle.textContent = message;
        } else {
            emptyState.innerHTML = `<p>${message}</p>`;
        }
    }
    emptyState.style.display = 'block';
    resultsContainer.style.display = 'none';
}

function hideEmptyState() {
    emptyState.style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
