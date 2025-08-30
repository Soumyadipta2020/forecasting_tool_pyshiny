// ===== Custom JavaScript for AI Forecasting App =====

$(document).ready(function() {
  // ----- Add icons to the navigation items -----
  // Home icon
  $('a.nav-link:contains("Home")').prepend('<i class="fa-solid fa-house"></i> ');
  
  // Forecasting icon
  $('a.nav-link:contains("Forecast")').prepend('<i class="fas fa-chart-line"></i> ');
  
  // About icon
  $('a.nav-link:contains("About")').prepend('<i class="fas fa-info-circle"></i> ');
  
  // ----- Add icons to tab navigation -----
  $('a.nav-link:contains("Data")').prepend('<i class="fas fa-database"></i> ');
  $('a.nav-link:contains("Summary Statistics")').prepend('<i class="fa-solid fa-chart-bar"></i>');
  
  // ----- Add custom header with logo -----
  var appHeader = `
    <nav class="navbar navbar-expand-lg navbar-dark">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">
          <i class="fas fa-brain"></i> AI Forecasting App <span class="version-badge" style='border-radius: 10px; font-size: small; background-color: #545454;'>&nbsp; v.0.0.1 &nbsp;</span>
        </a>
        <div class="ms-auto dark-mode-toggle">
          <!-- Dark mode toggle gets placed here by shiny -->
        </div>
      </div>
    </nav>
  `;
  
  // ----- Insert the custom header at the beginning of the body -----
  // We use setTimeout to ensure the DOM is fully loaded
  setTimeout(function() {
    $('body').prepend(appHeader);
    $('h2:contains("AI Forecasting App")').hide();
    
    // Move the dark mode toggle to the nav bar
    $('.dark-mode-toggle').append($('.float-end').html());
    $('.float-end').parent('.clearfix').hide();
  }, 500);
  
  // ----- Add a footer -----
  // var footer = `
  //   <footer class="footer mt-4">
  //     <div class="d-flex justify-content-between align-items-center">
  //       <div><i class="fas fa-copyright"></i> 2025</div>
  //       <div>Soumyadipta Das</div>
  //     </div>
  //   </footer>
  // `;
  
  // ----- Layout adjustments using Bootstrap classes -----
  setTimeout(function() {
    // Add the custom classes to main container
    $('.container-fluid').first().addClass('px-0');
    
    // Make the main content take full width
    $('.container-fluid > .row > div').first().removeClass('col-10').addClass('col-12');
    
    // Add footer to the appropriate location
    //$('.container-fluid').append(footer);
  }, 600);
  
  // ----- Card Collapse functionality -----
  // Listen for button click events for summary statistics section
  $(document).on('click', '#summary_collapse_btn', function() {
    var icon = $(this).find('i');
    var cardBody = $(this).closest('.card').find('.card-body');
    
    if (cardBody.is(':visible')) {
      cardBody.hide();
      icon.removeClass('fa-minus').addClass('fa-plus');
    } else {
      cardBody.show();
      icon.removeClass('fa-plus').addClass('fa-minus');
    }
  });
  
  // Listen for button click events for visualization section
  $(document).on('click', '#viz_collapse_btn', function() {
    var icon = $(this).find('i');
    var cardBody = $(this).closest('.card').find('.card-body');
    
    if (cardBody.is(':visible')) {
      cardBody.hide();
      icon.removeClass('fa-minus').addClass('fa-plus');
    } else {
      cardBody.show();
      icon.removeClass('fa-plus').addClass('fa-minus');
    }
  });
}); 